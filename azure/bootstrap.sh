#!/usr/bin/env bash
# Azure VM cloud-init (--custom-data). Idempotent: re-runs safely on every boot.
#
# Running on EVERY boot is not incidental. A spot VM created with --eviction-policy Delete is
# replaced rather than resumed, and local NVMe comes back RAW after any redeploy, so a
# first-boot-only setup leaves a later instance with no scratch mount and a confusing failure
# deep inside the run.
#
# The most expensive failure mode on Azure is that CLOUD-INIT FAILURE IS NOT PROVISIONING
# FAILURE: the portal reports the VM Running while this script died at step two, and a broken
# GPU VM bills at full rate for hours doing nothing. So this script publishes a heartbeat to
# blob storage at each stage, and the launcher reaps any VM that never heartbeats.
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/patriotsbreeze/SharpeningOrExpansion.git}"
BRANCH="${BRANCH:-claude/admiring-carson-k4s9to}"
WORKDIR="${WORKDIR:-/opt/soe}"
SCRATCH="${SCRATCH:-/mnt/soe}"
HEARTBEAT="/var/log/soe-heartbeat"

log() { echo "[bootstrap $(date -u +%H:%M:%SZ)] $*" | tee -a /var/log/soe-bootstrap.log; }

# Publish progress so the launcher can tell "still installing" from "silently dead".
beat() {
  echo "$1 $(date -u +%FT%TZ)" >> "${HEARTBEAT}"
  log "STAGE ${1}"
  if [[ -n "${AZ_CONTAINER_URL:-}" ]] && command -v azcopy >/dev/null 2>&1; then
    local name; name="$(hostname)"
    AZCOPY_AUTO_LOGIN_TYPE="${AZCOPY_AUTO_LOGIN_TYPE:-MSI}" \
      azcopy copy "${HEARTBEAT}" \
        "$(_url "heartbeat/${name}.log")" --overwrite=true --output-level=quiet 2>/dev/null || true
  fi
}
_url() {  # splice the path ahead of any SAS query string
  local sub="$1" base="${AZ_CONTAINER_URL}"
  if [[ "${base}" == *"?"* ]]; then echo "${base%%\?*}/${sub}?${base#*\?}";
  else echo "${base%/}/${sub}"; fi
}

trap 'log "BOOTSTRAP FAILED at line ${LINENO}"; beat failed' ERR

beat start

# ------------------------------------------------------------------ scratch on local NVMe
# The Ubuntu-HPC image does NOT ship /mnt/resource_nvme or an nvme-raid unit -- the call site
# exists but is commented out -- so we do it ourselves, idempotently.
log "preparing scratch"
mapfile -t NVME < <(lsblk -dpno NAME,TYPE | awk '$2=="disk" && $1 ~ /nvme/ {print $1}')
if ((${#NVME[@]} > 0)) && ! mountpoint -q "${SCRATCH}"; then
  mkdir -p "${SCRATCH}"
  if ((${#NVME[@]} == 1)); then
    DEV="${NVME[0]}"
  else
    DEV=/dev/md0
    if [[ ! -b "${DEV}" ]]; then
      log "striping ${#NVME[@]} NVMe devices into ${DEV}"
      mdadm --create "${DEV}" --level=0 --raid-devices="${#NVME[@]}" --run "${NVME[@]}" \
        || log "WARN: mdadm failed; falling back to ${NVME[0]}"
      [[ -b "${DEV}" ]] || DEV="${NVME[0]}"
    fi
  fi
  blkid "${DEV}" >/dev/null 2>&1 || mkfs.ext4 -F -m 0 "${DEV}"
  mount -o discard,noatime "${DEV}" "${SCRATCH}" || log "WARN: could not mount ${DEV}"
fi
# Fall back to the SCSI temp disk, then the OS disk. Never fail here -- a slower cache is
# far better than an unbootstrapped GPU.
mountpoint -q "${SCRATCH}" || { mkdir -p /mnt/soe && SCRATCH=/mnt/soe; }
mkdir -p "${SCRATCH}/hf" "${SCRATCH}/runs"
chmod 777 "${SCRATCH}" "${SCRATCH}/hf" "${SCRATCH}/runs"
log "scratch at ${SCRATCH} ($(df -h --output=avail "${SCRATCH}" | tail -1 | tr -d ' ') free)"
beat scratch

# ------------------------------------------------------------------ eviction watcher, at boot
# Scheduled Events is lazily enabled: the first call can take ~2 minutes, events only flow
# after ~5, and the endpoint self-disables after 24h without polling. A watcher started when
# the JOB starts is therefore blind for exactly the window in which an early eviction would
# cost the most, so it starts here instead and just records notices to a sentinel file.
#
# This only ever GETs. Acknowledging a Preempt event by POSTing tells Azure to proceed
# immediately and forfeits the remaining notice window.
cat > /usr/local/bin/soe-preempt-watch <<'WATCH'
#!/usr/bin/env bash
URL="http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01"
SENTINEL=/var/run/soe_preempt
while true; do
  body=$(curl -s -m 5 -H Metadata:true "${URL}" 2>/dev/null || true)
  if [[ "${body}" == *Preempt* || "${body}" == *Terminate* ]]; then
    printf '%s %s\n' "$(date -u +%FT%TZ)" "${body}" > "${SENTINEL}"
  fi
  sleep 1
done
WATCH
chmod +x /usr/local/bin/soe-preempt-watch
cat > /etc/systemd/system/soe-preempt-watch.service <<'UNIT'
[Unit]
Description=Poll Azure Scheduled Events for spot eviction notices
[Service]
ExecStart=/usr/local/bin/soe-preempt-watch
Restart=always
RestartSec=2
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now soe-preempt-watch
beat watcher

# ------------------------------------------------------------------ tooling
log "verifying GPU stack (the image ships the driver; we do not install one)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader \
  || log "WARN: nvidia-smi failed -- wrong image? expected microsoft-dsvm:ubuntu-hpc"
[[ -f /opt/azurehpc/component_versions.txt ]] && \
  log "image components: $(tr '\n' ' ' < /opt/azurehpc/component_versions.txt)"

command -v azcopy >/dev/null 2>&1 || {
  log "installing azcopy"
  curl -sL https://aka.ms/downloadazcopy-v10-linux -o /tmp/azcopy.tgz
  tar -xzf /tmp/azcopy.tgz -C /tmp
  install -m 0755 /tmp/azcopy_linux_amd64_*/azcopy /usr/local/bin/azcopy
}
log "azcopy $(azcopy --version 2>/dev/null | head -1)"
beat tooling

# ------------------------------------------------------------------ repo + deps
log "cloning ${REPO_URL}@${BRANCH}"
if [[ -d "${WORKDIR}/.git" ]]; then
  git -C "${WORKDIR}" fetch --depth 1 origin "${BRANCH}"
  git -C "${WORKDIR}" checkout -f FETCH_HEAD
else
  git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
fi
cd "${WORKDIR}"
beat clone

log "installing dependencies"
export HOME=/root
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:${PATH}"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gpu,dev]"
beat deps

cat > /etc/profile.d/soe.sh <<EOF
export HF_HOME=${SCRATCH}/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export SOE_ROOT=${SCRATCH}/runs
export SOE_OBJSTORE=azure
export AZ_CONTAINER_URL='${AZ_CONTAINER_URL:-}'
export AZCOPY_AUTO_LOGIN_TYPE=${AZCOPY_AUTO_LOGIN_TYPE:-MSI}
export SOE_WORKER='${SOE_WORKER:-}'
export PATH=${WORKDIR}/.venv/bin:/root/.local/bin:\$PATH
export PYTHONPATH=${WORKDIR}/src
EOF
# shellcheck disable=SC1091
source /etc/profile.d/soe.sh

log "preflight"
"${WORKDIR}/.venv/bin/python" -m soe.cli registry check
beat ready

# ------------------------------------------------------------------ optional autostart
if [[ -n "${SOE_CONFIG:-}" ]]; then
  log "autostarting ${SOE_CONFIG} as worker ${SOE_WORKER:-<local GPUs>}"
  nohup bash "${WORKDIR}/scripts/launch_node.sh" "${SOE_CONFIG}" \
    > /var/log/soe-run.log 2>&1 &
  beat running
else
  log "ready. ssh in, source /etc/profile.d/soe.sh, then scripts/launch_node.sh <config>"
fi
