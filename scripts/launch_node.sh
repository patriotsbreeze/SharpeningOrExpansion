#!/usr/bin/env bash
# Run generation on this machine, then grade and verify.
#
# Two topologies, same script:
#
#   FLEET MODE   (SOE_WORKER set)  -- this VM runs exactly ONE worker with that global index.
#                                     Used for a fleet of single-GPU VMs, which is the cheaper
#                                     and far more obtainable shape on Azure for a workload
#                                     that needs no inter-GPU communication.
#   NODE MODE    (SOE_WORKER unset) -- one worker per local GPU on a multi-GPU box.
#
# The worker index is global either way, and it must be < n_workers in the experiment config.
# n_workers determines the problem partition, so it is a property of the experiment, not of
# whatever hardware happens to be available. Changing it against an existing artifact tree
# reassigns problems to different shards while leaving seeds unchanged, which writes the same
# sample under two paths.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/objstore.sh
source "${HERE}/objstore.sh"

CONFIG="${1:?usage: launch_node.sh <config.yaml> [n_local_gpus]}"
ROOT="${SOE_ROOT:-runs}"
STAGGER="${STAGGER:-30}"
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
GRADERS="${GRADERS:-fastint,mathverify}"

EXP=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['exp_id'])" "${CONFIG}")
NWORKERS=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['n_workers'])" "${CONFIG}")
EXPDIR="${ROOT}/exp=${EXP}"
LOGDIR="${EXPDIR}/logs"
mkdir -p "${LOGDIR}"

if [[ -n "${SOE_WORKER:-}" ]]; then
  MODE=fleet
  WORKERS=("${SOE_WORKER}")
else
  MODE=node
  NGPU="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
  [[ "${NGPU}" -gt 0 ]] || { echo "[launch] no GPUs detected and SOE_WORKER unset" >&2; exit 1; }
  WORKERS=()
  for ((g = 0; g < NGPU; g++)); do WORKERS+=("${g}"); done
fi

for w in "${WORKERS[@]}"; do
  if (( w < 0 || w >= NWORKERS )); then
    echo "[launch] worker index ${w} is outside the configured fleet of ${NWORKERS}." >&2
    echo "[launch] Valid: 0..$((NWORKERS - 1)). Change n_workers in ${CONFIG} only for a" >&2
    echo "[launch] fresh artifact tree -- it repartitions problems across shards." >&2
    exit 2
  fi
done
echo "[launch] exp=${EXP} mode=${MODE} workers=[${WORKERS[*]}] of ${NWORKERS} root=${ROOT}"

# ---------------------------------------------------------------- preemption watcher
# Azure Spot gives roughly 30 seconds of notice through the Scheduled Events metadata
# endpoint, and an eviction is not guaranteed to arrive as a catchable signal -- so polling
# the endpoint is what actually saves the last sync interval of work, not a bash trap alone.
watch_for_eviction() {
  local url="http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01"
  while true; do
    local body
    body=$(curl -s -m 5 -H Metadata:true "${url}" 2>/dev/null || true)
    if [[ "${body}" == *'"EventType":"Preempt"'* || "${body}" == *'"EventType": "Preempt"'* ]]; then
      echo "[launch] PREEMPT notice received -- final sync" | tee -a "${LOGDIR}/eviction.log"
      objstore_push "${EXPDIR}" "exp=${EXP}" || true
      echo "[launch] final sync done" | tee -a "${LOGDIR}/eviction.log"
      return 0
    fi
    sleep 5
  done
}

# ---------------------------------------------------------------- resume + manifest
if objstore_configured; then
  echo "[launch] pulling completion markers (not shards) to reconstruct progress"
  objstore_pull_markers "exp=${EXP}" "${EXPDIR}" || echo "[launch] no prior markers"
fi

# Every worker plans independently from the same config, so all machines must agree on the
# problem manifest -- problem_idx feeds the seed, so a manifest that differs between machines
# makes the same seed denote different problems. The lead worker publishes one manifest and
# the rest consume it, which removes that class of failure rather than detecting it later.
LEAD=0
have_manifest() { compgen -G "${EXPDIR}/problems/*.manifest.jsonl" > /dev/null; }

if objstore_configured; then
  objstore_pull_dir "exp=${EXP}/problems" "${EXPDIR}/problems" 2>/dev/null || true
fi

if ! have_manifest; then
  if [[ " ${WORKERS[*]} " == *" ${LEAD} "* ]] || ! objstore_configured; then
    echo "[launch] preparing problem manifests (lead worker)"
    python -m soe.cli prepare "${CONFIG}" --root "${ROOT}"
    if objstore_configured; then
      for m in "${EXPDIR}"/problems/*.manifest.jsonl; do
        objstore_push_file "${m}" "exp=${EXP}/problems/$(basename "${m}")"
      done
    fi
  else
    echo "[launch] waiting for the lead worker to publish manifests"
    for _ in $(seq 1 60); do
      sleep 10
      objstore_pull_dir "exp=${EXP}/problems" "${EXPDIR}/problems" 2>/dev/null || true
      have_manifest && break
    done
    have_manifest || {
      echo "[launch] manifests never appeared. Refusing to build our own: two machines with" >&2
      echo "[launch] different manifests corrupt the run silently. Start worker ${LEAD} first." >&2
      exit 3
    }
  fi
fi

python -m soe.cli plan "${CONFIG}" --root "${ROOT}"

# ---------------------------------------------------------------- background sync
SYNC_PID=""
WATCH_PID=""
if objstore_configured; then
  ( while true; do sleep "${SYNC_INTERVAL}"; objstore_push "${EXPDIR}" "exp=${EXP}" || true; done ) &
  SYNC_PID=$!
  watch_for_eviction & WATCH_PID=$!
fi
cleanup() {
  [[ -n "${SYNC_PID}" ]] && kill "${SYNC_PID}" 2>/dev/null || true
  [[ -n "${WATCH_PID}" ]] && kill "${WATCH_PID}" 2>/dev/null || true
  objstore_configured && objstore_push "${EXPDIR}" "exp=${EXP}" || true
}
trap cleanup EXIT TERM INT

# ---------------------------------------------------------------- generate
pids=()
for w in "${WORKERS[@]}"; do
  if [[ "${MODE}" == node ]]; then
    export CUDA_VISIBLE_DEVICES="${w}"
  fi
  python -m soe.cli generate "${CONFIG}" --root "${ROOT}" --worker "${w}" \
      > "${LOGDIR}/worker${w}.log" 2>&1 &
  pids+=($!)
  echo "[launch] worker ${w} -> pid ${pids[-1]}"
  # N x ~15GB of weights loading at once spikes host RAM well past what the VM has.
  [[ ${#WORKERS[@]} -gt 1 ]] && sleep "${STAGGER}"
done

fail=0
for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
echo "[launch] generation finished (fail=${fail})"

# ---------------------------------------------------------------- grade + verify
python -m soe.cli grade  "${CONFIG}" --root "${ROOT}" --graders "${GRADERS}"
python -m soe.cli verify "${CONFIG}" --root "${ROOT}" --no-require-complete || true
exit "${fail}"
