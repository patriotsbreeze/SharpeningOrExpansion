#!/usr/bin/env bash
# Delete the whole run. This is the ONLY complete teardown.
#
# `az vm delete` does not cascade: it leaves the public IP, NIC and managed disks behind, and
# an unattached Standard public IP and an orphaned disk both keep billing indefinitely after
# the run has "ended". Deleting the resource group is what actually stops the meter.
set -Eeuo pipefail
RG="${RG:-soe-run}"

echo "Resources in ${RG}:"
az resource list -g "${RG}" --query "[].{name:name,type:type}" -o table 2>/dev/null || {
  echo "resource group ${RG} not found -- nothing to tear down"; exit 0; }

read -r -p "Delete resource group '${RG}' and everything in it? [y/N] " ok
[[ "${ok}" == [yY] ]] || { echo "aborted"; exit 1; }

az group delete -n "${RG}" --yes
echo "deleted ${RG}"
echo "NOTE: the storage account holding the artifacts is deliberately in a DIFFERENT group,"
echo "      so your results survive this. Verify before you rely on it:"
echo "      az storage account list --query \"[].{n:name,rg:resourceGroup}\" -o table"
