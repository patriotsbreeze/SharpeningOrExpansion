#!/usr/bin/env bash
# EC2 user-data for a Deep Learning AMI node. Idempotent: safe to re-run after a spot reclaim.
set -Eeuo pipefail

REPO_URL="${REPO_URL:-https://github.com/patriotsbreeze/SharpeningOrExpansion.git}"
BRANCH="${BRANCH:-claude/admiring-carson-k4s9to}"
WORKDIR="${WORKDIR:-/opt/soe}"
# Instance-store NVMe. Model weights are tens of GB and the root EBS volume is both smaller
# and much slower; downloading 6 checkpoints to EBS is a silent half-hour tax per boot.
NVME="${NVME:-/opt/dlami/nvme}"

log() { echo "[bootstrap $(date -u +%H:%M:%S)] $*"; }

log "preparing scratch on ${NVME}"
if [[ ! -d "${NVME}" ]]; then
  DEV=$(lsblk -dpno NAME,TYPE | awk '$2=="disk"{print $1}' | tail -1)
  mkdir -p "${NVME}"
  mkfs.ext4 -F "${DEV}" >/dev/null 2>&1 || true
  mount "${DEV}" "${NVME}" || log "WARN: could not mount ${DEV}; falling back to root volume"
fi
mkdir -p "${NVME}/hf" "${NVME}/runs"
export HF_HOME="${NVME}/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1

log "cloning ${REPO_URL}@${BRANCH}"
if [[ -d "${WORKDIR}/.git" ]]; then
  git -C "${WORKDIR}" fetch --depth 1 origin "${BRANCH}" && git -C "${WORKDIR}" checkout -f FETCH_HEAD
else
  git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${WORKDIR}"
fi
cd "${WORKDIR}"

log "installing dependencies"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e ".[gpu,dev]"

cat > /etc/profile.d/soe.sh <<EOF
export HF_HOME=${NVME}/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export SOE_ROOT=${NVME}/runs
export PATH=${WORKDIR}/.venv/bin:\$PATH
export PYTHONPATH=${WORKDIR}/src
EOF

log "preflight"
"${WORKDIR}/.venv/bin/python" -m soe.cli registry check
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || log "WARN: no GPU visible"

log "ready. Next: aws/README.md step 4"
