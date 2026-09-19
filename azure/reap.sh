#!/usr/bin/env bash
# Delete VMs that are billing but not working.
#
# This is the counterpart to bootstrap.sh's heartbeat, and it has to exist as a real command
# rather than as advice in a comment: the most expensive failure mode on Azure is that
# CLOUD-INIT FAILURE IS NOT PROVISIONING FAILURE. The portal reports the VM Running while
# bootstrap died at step two, and an idle NC40ads_H100_v5 bills at full rate for as long as
# nobody notices.
#
# Two conditions are reaped:
#   never-started : no heartbeat blob at all after --deadline minutes
#   stale         : last beat older than --stale minutes (bootstrap beats `alive` every 60s
#                   while the job runs, and `job-exit <rc>` when it finishes)
#
# A VM whose last beat is `job-exit 0` finished successfully and is reaped too -- its work is
# already in blob storage, so keeping it alive is pure cost.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/azwrap.sh
source "${HERE}/../scripts/azwrap.sh"
# shellcheck source=scripts/objstore.sh
source "${HERE}/../scripts/objstore.sh"

RG="${RG:-soe-run}"
AZ_CONTAINER_URL="${AZ_CONTAINER_URL:?set AZ_CONTAINER_URL}"
DEADLINE_MIN="${DEADLINE_MIN:-25}"     # generous: bootstrap installs torch and vllm
STALE_MIN="${STALE_MIN:-15}"
DRY_RUN="${DRY_RUN:-1}"                # default to reporting; set DRY_RUN=0 to delete

command -v azcopy >/dev/null || { echo "azcopy not found" >&2; exit 1; }
now=$(date -u +%s)
work=$(mktemp -d); trap 'rm -rf "${work}"' EXIT

# _az_url comes from objstore.sh. The private copy this file used to carry duplicated the
# SAS-splicing logic that the objstore tests cover, so a fix in one never reached the other.

azcopy copy "$(_az_url "heartbeat" "/*")" "${work}" --recursive --overwrite=true \
  --output-level=quiet 2>/dev/null || true

mapfile -t VMS < <(azq vm list -g "${RG}" --query "[].name" -o tsv 2>/dev/null || true)
((${#VMS[@]})) || { echo "no VMs in ${RG}"; exit 0; }

reap=()
for vm in "${VMS[@]}"; do
  created=$(azq vm show -g "${RG}" -n "${vm}" --query "timeCreated" -o tsv 2>/dev/null || echo "")
  age_min=9999
  if [[ -n "${created}" ]]; then
    age_min=$(( (now - $(date -u -d "${created}" +%s)) / 60 ))
  fi

  beat_file=$(find "${work}" -name "${vm}.log" -print -quit 2>/dev/null || true)
  if [[ -z "${beat_file}" ]]; then
    if (( age_min > DEADLINE_MIN )); then
      echo "REAP ${vm}: no heartbeat after ${age_min}m (bootstrap never got far enough)"
      reap+=("${vm}")
    else
      echo "wait ${vm}: no heartbeat yet, ${age_min}m old (deadline ${DEADLINE_MIN}m)"
    fi
    continue
  fi

  last=$(tail -1 "${beat_file}")
  stamp=$(awk '{print $NF}' <<<"${last}")
  beat_age=$(( (now - $(date -u -d "${stamp}" +%s 2>/dev/null || echo "${now}")) / 60 ))
  stage=$(awk '{print $1}' <<<"${last}")

  if [[ "${stage}" == "job-exit" ]]; then
    echo "REAP ${vm}: job finished (${last})"
    reap+=("${vm}")
  elif (( beat_age > STALE_MIN )); then
    echo "REAP ${vm}: last beat '${stage}' was ${beat_age}m ago"
    reap+=("${vm}")
  else
    echo "ok   ${vm}: '${stage}' ${beat_age}m ago"
  fi
done

((${#reap[@]})) || { echo "nothing to reap"; exit 0; }
if [[ "${DRY_RUN}" != "0" ]]; then
  echo ""
  echo "DRY RUN. Re-run with DRY_RUN=0 to delete: ${reap[*]}"
  exit 0
fi
for vm in "${reap[@]}"; do
  echo "deleting ${vm}"
  azrun vm delete -g "${RG}" -n "${vm}" --yes --no-wait
done
