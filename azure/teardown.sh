#!/usr/bin/env bash
# Delete the whole run. This is the ONLY complete teardown.
#
# `az vm delete` does not cascade: it leaves the public IP, NIC and managed disks behind, and
# an unattached Standard public IP and an orphaned disk both keep billing indefinitely after
# the run has "ended". Deleting the resource group is what actually stops the meter.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/azwrap.sh
source "${HERE}/../scripts/azwrap.sh"

RG="${RG:-soe-run}"
FORCE="${FORCE:-0}"

echo "Resources in ${RG}:"
az_banner
# Distinguish "the group is gone" from "az failed": reporting a failed call as nothing-to-do
# leaves live GPU VMs billing while the operator believes they are deleted.
if ! LISTING=$(azq resource list -g "${RG}" --query "[].{name:name,type:type}" -o table 2>&1); then
  if [[ "${LISTING}" == *"ResourceGroupNotFound"* ]]; then
    echo "resource group ${RG} not found -- nothing to tear down"
    exit 0
  fi
  echo "could not list ${RG}; NOT assuming it is empty:" >&2
  echo "${LISTING}" >&2
  exit 1
fi
echo "${LISTING}"

# An interactive prompt blocks forever under CI or subprocess capture, so FORCE=1 is the
# non-interactive path rather than a missing stdin being treated as a "no".
if [[ "${FORCE}" != "1" ]] && ! az_dry; then
  if [[ ! -t 0 ]]; then
    echo "not a terminal; re-run with FORCE=1 to confirm non-interactively" >&2
    exit 1
  fi
  read -r -p "Delete resource group '${RG}' and everything in it? [y/N] " ok
  [[ "${ok}" == [yY] ]] || { echo "aborted"; exit 1; }
fi

azrun group delete -n "${RG}" --yes
echo "deleted ${RG}"
echo "NOTE: the storage account holding the artifacts is deliberately in a DIFFERENT group,"
echo "      so your results survive this. Verify before you rely on it:"
echo "      az storage account list --query \"[].{n:name,rg:resourceGroup}\" -o table"
