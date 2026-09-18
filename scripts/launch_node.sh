#!/usr/bin/env bash
# One vLLM worker per GPU, each pinned to its own device.
#
# TP=1 with N independent replicas, sharding PROBLEMS across workers, rather than
# tensor-parallel across GPUs: a 7B in bf16 is ~15 GB, so TP only adds collective overhead.
#
# Workers are staggered because N x 15 GB of weights loading simultaneously spikes host RAM
# far above what the instance has, and the OOM killer takes the whole run with it.
set -Eeuo pipefail

CONFIG="${1:?usage: launch_node.sh <config.yaml> [n_gpus]}"
NGPU="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
ROOT="${SOE_ROOT:-runs}"
S3_URI="${S3_URI:-}"
STAGGER="${STAGGER:-30}"
LOGDIR="${ROOT}/logs"
mkdir -p "${LOGDIR}"

EXP=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['exp_id'])" "${CONFIG}")
echo "[launch] exp=${EXP} gpus=${NGPU} root=${ROOT}"

# Pull only markers (kilobytes) to reconstruct what is already done. Never pull the shards.
if [[ -n "${S3_URI}" ]]; then
  echo "[launch] syncing completion markers from ${S3_URI}"
  aws s3 sync "${S3_URI}/exp=${EXP}/" "${ROOT}/exp=${EXP}/" \
      --exclude "*" --include "*.done.json" --only-show-errors
fi

python -m soe.cli prepare "${CONFIG}" --root "${ROOT}"
python -m soe.cli plan    "${CONFIG}" --root "${ROOT}"

if [[ -n "${S3_URI}" ]]; then
  ( while true; do
      aws s3 sync "${ROOT}/exp=${EXP}/" "${S3_URI}/exp=${EXP}/" --only-show-errors || true
      sleep 300
    done ) &
  SYNC_PID=$!
  # Final sync on the way out -- including on a spot reclaim, which arrives as SIGTERM.
  trap 'kill ${SYNC_PID} 2>/dev/null || true; \
        aws s3 sync "${ROOT}/exp=${EXP}/" "${S3_URI}/exp=${EXP}/" --only-show-errors || true' EXIT TERM INT
fi

pids=()
for ((g=0; g<NGPU; g++)); do
  CUDA_VISIBLE_DEVICES="${g}" python -m soe.cli generate "${CONFIG}" \
      --root "${ROOT}" --worker "${g}" > "${LOGDIR}/worker${g}.log" 2>&1 &
  pids+=($!)
  echo "[launch] worker ${g} -> pid ${pids[-1]}"
  sleep "${STAGGER}"
done

fail=0
for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
echo "[launch] generation finished (fail=${fail})"

python -m soe.cli grade  "${CONFIG}" --root "${ROOT}" --graders fastint,mathverify
python -m soe.cli verify "${CONFIG}" --root "${ROOT}" --no-require-complete
exit "${fail}"
