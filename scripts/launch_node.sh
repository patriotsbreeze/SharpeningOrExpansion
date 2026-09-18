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

# ---------------------------------------------------------------- preemption reaction
# The DETECTION half lives in azure/bootstrap.sh, which starts a systemd poller at boot and
# writes any notice to a sentinel file. That split matters: Scheduled Events is lazily
# enabled, so the endpoint needs ~2 minutes before it answers and ~5 before events flow, and
# it self-disables after 24h without polling. A watcher that only started when this script
# ran would be blind through exactly the window where an early eviction costs the most.
#
# Here we only REACT: notice the sentinel and get the artifacts out. Azure gives roughly 30
# seconds, and an eviction is not guaranteed to arrive as a catchable signal, so the trap
# below is a backstop rather than the mechanism.
PREEMPT_SENTINEL="${PREEMPT_SENTINEL:-/var/run/soe_preempt}"

react_to_eviction() {
  while true; do
    if [[ -f "${PREEMPT_SENTINEL}" ]]; then
      echo "[launch] eviction notice seen -- final sync"
      cat "${PREEMPT_SENTINEL}" 2>/dev/null || true
      objstore_push "${EXPDIR}" "exp=${EXP}" || true
      echo "[launch] final sync complete"
      return 0
    fi
    sleep 1
  done
}

# ---------------------------------------------------------------- resume + manifest
if objstore_configured; then
  echo "[launch] pulling completion markers (not shards) to reconstruct progress"
  if ! objstore_pull_markers "exp=${EXP}" "${EXPDIR}"; then
    # Distinguishing "nothing published yet" from "could not reach the store" matters: the
    # latter looks like a fresh run, so the worker would regenerate everything already done
    # and then write duplicate samples over the top of it.
    echo "[launch] could not read completion markers from the object store." >&2
    echo "[launch] Continuing would re-generate completed chunks and duplicate samples." >&2
    exit 4
  fi
fi

# Every worker plans independently from the same config, so all machines must agree on the
# problem manifest: problem_idx is a position in that file and feeds every seed, so manifests
# that differ between machines make the same seed denote different problems.
#
# Leadership is NOT tied to index 0. Partial capacity is the expected case for GPU spot, and
# azure/launch.sh deliberately proceeds when some VMs fail to create -- so if index 0 is the
# one Azure declined, a fixed lead would leave every other VM polling until it gave up, having
# produced nothing while billing for GPUs the whole time. SOE_LEAD is set by the launcher to
# the lowest index it ACTUALLY created; absent that, the lowest index on this machine.
#
# Leadership is also not exclusive. If the lead never publishes -- evicted during bootstrap, or
# its dataset download outran the wait -- any worker may take over once the deadline passes.
# That is safe because publication is write-once: whoever wins, everyone then consumes the
# winner's bytes rather than their own, so the fleet cannot split across two manifests.
LEAD="${SOE_LEAD:-${WORKERS[0]}}"
MANIFEST_WAIT="${MANIFEST_WAIT:-60}"   # x10s

# Which datasets this experiment actually needs -- an existential "any manifest present" check
# would let a worker proceed on a partially published directory and then fail deep in the run.
mapfile -t DATASETS < <(python - "${CONFIG}" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
print("\n".join(sorted({a["dataset_key"] for a in cfg["arms"]})))
PYEOF
)

have_manifest() {
  local k
  for k in "${DATASETS[@]}"; do
    [[ -s "${EXPDIR}/problems/${k}.manifest.jsonl" ]] || return 1
  done
  return 0
}

publish_manifests() {
  local m rc=0
  for k in "${DATASETS[@]}"; do
    m="${EXPDIR}/problems/${k}.manifest.jsonl"
    [[ -s "${m}" ]] || { echo "[launch] prepare produced no manifest for ${k}" >&2; return 1; }
    objstore_publish_once "${m}" "exp=${EXP}/problems/${k}.manifest.jsonl" || rc=1
  done
  return "${rc}"
}

pull_manifests() {
  objstore_configured || return 0
  objstore_pull_dir "exp=${EXP}/problems" "${EXPDIR}/problems" 2>/dev/null || true
}

pull_manifests
if ! have_manifest; then
  if [[ "${WORKERS[0]}" == "${LEAD}" ]] || ! objstore_configured; then
    echo "[launch] lead worker (${LEAD}); preparing problem manifests"
    python -m soe.cli prepare "${CONFIG}" --root "${ROOT}"
    publish_manifests || echo "[launch] WARN: publish incomplete; will re-pull the winner's copy"
  else
    echo "[launch] waiting up to $((MANIFEST_WAIT * 10))s for worker ${LEAD} to publish"
    for _ in $(seq 1 "${MANIFEST_WAIT}"); do
      sleep 10
      pull_manifests
      have_manifest && break
    done
    if ! have_manifest; then
      # Takeover. Safe only because publication is write-once.
      echo "[launch] lead ${LEAD} never published; taking over (publication is write-once)"
      python -m soe.cli prepare "${CONFIG}" --root "${ROOT}"
      publish_manifests || true
    fi
  fi

  # Always re-pull and use the PUBLISHED copy, even if we prepared it ourselves. If another
  # worker won the write-once race, its bytes are authoritative and ours must be discarded --
  # otherwise two machines proceed on two different problem orderings.
  if objstore_configured; then
    rm -f "${EXPDIR}"/problems/*.manifest.jsonl
    for _ in 1 2 3; do
      pull_manifests
      have_manifest && break
      sleep 5
    done
  fi

  have_manifest || {
    echo "[launch] no usable problem manifest for: ${DATASETS[*]}" >&2
    echo "[launch] Refusing to run: proceeding on a manifest other machines do not share" >&2
    echo "[launch] would make identical seeds denote different problems." >&2
    exit 3
  }
fi

python -m soe.cli plan "${CONFIG}" --root "${ROOT}"

# ---------------------------------------------------------------- background sync
SYNC_PID=""
WATCH_PID=""
if objstore_configured; then
  # Prove the object store is actually usable BEFORE burning GPU hours. Every objstore_push
  # failure downstream is tolerated so a transient blip does not kill a run, which means
  # without this probe a misconfigured container would let the whole run complete and then
  # vanish with the VM.
  objstore_require || exit 4
  probe="${EXPDIR}/.objstore_probe"
  mkdir -p "${EXPDIR}"
  date -u +%FT%TZ > "${probe}"
  if ! objstore_push "${EXPDIR}" "exp=${EXP}"; then
    echo "[launch] object store is not writable; refusing to start a run that cannot persist" >&2
    exit 4
  fi
  rm -f "${probe}"

  # Both helpers redirect their own output. A background job that inherits this script's
  # stdout keeps the pipe open after the script exits, so any caller that captures output --
  # a CI harness, `az vm run-command`, subprocess.run(capture_output=True) -- blocks until
  # the helper happens to die rather than when the run actually finishes.
  ( fails=0
    while true; do
      sleep "${SYNC_INTERVAL}"
      if objstore_push "${EXPDIR}" "exp=${EXP}"; then
        fails=0
      else
        fails=$((fails + 1))
        echo "[sync] push failed (${fails} consecutive)"
        # Tolerate a blip, but a store that stays unreachable means everything produced from
        # here on dies with the VM. Stop rather than keep burning GPU hours undurably.
        if (( fails >= 3 )); then
          echo "[sync] giving up after ${fails} consecutive failures; signalling the run"
          touch "${EXPDIR}/.sync_broken"
          exit 1
        fi
      fi
    done ) >> "${LOGDIR}/sync.log" 2>&1 &
  SYNC_PID=$!
  react_to_eviction >> "${LOGDIR}/eviction.log" 2>&1 &
  WATCH_PID=$!
fi
cleanup() {
  # Reap the helpers FIRST, so the final push does not race the periodic one and nothing is
  # left holding an inherited file descriptor.
  local pid
  for pid in "${SYNC_PID}" "${WATCH_PID}"; do
    [[ -n "${pid}" ]] || continue
    # Kill the descendants too. The syncer spends almost all its life blocked in `sleep`, and
    # killing only the subshell leaves that sleep alive holding any inherited descriptor.
    pkill -P "${pid}" 2>/dev/null || true
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  done
  SYNC_PID="" ; WATCH_PID=""
  if objstore_configured; then
    objstore_push "${EXPDIR}" "exp=${EXP}" || true
  fi
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
[[ -f "${EXPDIR}/.sync_broken" ]] && { echo "[launch] object store became unreachable mid-run" >&2; fail=1; }

# ---------------------------------------------------------------- grade + verify
python -m soe.cli grade  "${CONFIG}" --root "${ROOT}" --graders "${GRADERS}"

# --no-deep is required here, not a shortcut. Resume pulls only the MARKERS from blob
# storage, never the shards, so on any resumed VM most shards this machine has a marker for
# are not on its local disk. A deep verify reads every shard to check its checksum and would
# report a failure for each absent one -- drowning any real problem in noise.
#
# The full deep verify belongs on the analysis machine, once, against the complete tree
# pulled down from blob storage (azure/README.md step 8). That is also the only place where
# the cross-machine checks -- global seed uniqueness, equal n per problem, one problem_uid
# per problem index -- can actually see the whole run.
python -m soe.cli verify "${CONFIG}" --root "${ROOT}" --no-deep --no-require-complete || true
exit "${fail}"
