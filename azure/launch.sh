#!/usr/bin/env bash
# Create a fleet of single-GPU Azure Spot VMs, one worker each.
#
# WHY A LOOP AND NOT A SCALE SET. Uniform VMSS is the only mode with numeric instance names,
# and `az vmss create` now defaults to Flexible, whose names are {vmss}_{8-char-guid} -- no
# index at all. Even under Uniform, Azure explicitly declines to guarantee instance IDs are
# 0..N-1, and IDs are reused after deletion. Since a worker's index selects its problem shard,
# a loop that passes the index explicitly is the only way to be sure who owns what.
#
# The harness tolerates a wrong index anyway -- markers make chunks idempotent and the steal
# pass covers any shard nobody claimed -- but a collision wastes GPU time, and GPU time is the
# scarce resource here.
#
# WHY SINGLE-GPU VMs. This workload runs tensor_parallel_size=1 with one engine per GPU and no
# inter-worker communication, so an 8-GPU node buys nothing. It also concentrates risk: one
# eviction takes all eight workers. And ND-series 8-GPU nodes bust the budget at
# pay-as-you-go rates, while single-GPU NC-series does not.
set -Eeuo pipefail

# Prefer the H100 NC line: Azure states it is only deploying net new capacity for
# NCads_H100_v5, so the A100 pool is frozen and will only get harder to allocate.
VM_SIZE="${VM_SIZE:-Standard_NC40ads_H100_v5}"
LOCATION="${LOCATION:?set LOCATION, e.g. eastus2 -- pick it on eviction rate, see README step 4}"
RG="${RG:-soe-run}"
FLEET="${FLEET:-4}"
CONFIG="${CONFIG:-configs/experiments/stage1_tierA.yaml}"
AZ_CONTAINER_URL="${AZ_CONTAINER_URL:?set AZ_CONTAINER_URL to the blob container URL}"
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:?set STORAGE_ACCOUNT (needed to grant the identity access)}"
# microsoft-dsvm:ubuntu-hpc ships the NVIDIA driver, CUDA and Fabric Manager. A stock
# Canonical image has NO driver, so the VM would bill at full rate while nvidia-smi fails.
IMAGE="${IMAGE:-microsoft-dsvm:ubuntu-hpc:2404:latest}"
ADMIN="${ADMIN:-azureuser}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa.pub}"
PRIORITY="${PRIORITY:-Spot}"
OS_DISK_GB="${OS_DISK_GB:-256}"
BRANCH="${BRANCH:-claude/admiring-carson-k4s9to}"

command -v az >/dev/null || { echo "az CLI not found" >&2; exit 1; }
[[ -f "${SSH_KEY}" ]] || { echo "no ssh public key at ${SSH_KEY}" >&2; exit 1; }

echo "[launch] ${FLEET}x ${VM_SIZE} (${PRIORITY}) in ${LOCATION}, rg=${RG}"

# A dedicated resource group is the teardown story: `az vm delete` does not cascade, and
# orphaned public IPs and disks keep billing after the run "ended".
az group create -n "${RG}" -l "${LOCATION}" -o none
echo "[launch] resource group ${RG} ready (delete it to tear everything down)"

SUB=$(az account show --query id -o tsv)
SCOPE="/subscriptions/${SUB}/resourceGroups/$(az storage account show \
  --name "${STORAGE_ACCOUNT}" --query resourceGroup -o tsv)/providers/Microsoft.Storage/storageAccounts/${STORAGE_ACCOUNT}"

created=() failed=()
for ((w = 0; w < FLEET; w++)); do
  NAME="soe-w${w}"
  USERDATA=$(mktemp)
  {
    echo '#!/usr/bin/env bash'
    echo "export AZ_CONTAINER_URL='${AZ_CONTAINER_URL}'"
    echo "export SOE_WORKER='${w}'"
    echo "export SOE_CONFIG='${CONFIG}'"
    echo "export BRANCH='${BRANCH}'"
    echo "export AZCOPY_AUTO_LOGIN_TYPE=MSI"
    cat "$(dirname "$0")/bootstrap.sh"
  } > "${USERDATA}"

  echo "[launch] creating ${NAME} (worker ${w})"
  # --eviction-policy Delete, not the Deallocate default: a deallocated spot VM keeps billing
  #   for its disks AND keeps consuming spot quota, so replacements cannot allocate and the
  #   fleet strangles itself within hours.
  # --max-price -1 means "never evicted for price reasons, never charged above pay-as-you-go".
  #   (learn.microsoft.com's spot-cli page states the opposite; its spot-vms page states this
  #   twice and matches the pricing table. The CLI page has a dropped negation.)
  # --ephemeral-os-disk requires Delete eviction, which we already use. Temp-disk placement
  #   keeps the local NVMe free for the HF cache.
  if az vm create \
      --resource-group "${RG}" --name "${NAME}" --location "${LOCATION}" \
      --size "${VM_SIZE}" --image "${IMAGE}" \
      --admin-username "${ADMIN}" --ssh-key-values "${SSH_KEY}" \
      --priority "${PRIORITY}" --eviction-policy Delete --max-price -1 \
      --assign-identity '[system]' \
      --os-disk-size-gb "${OS_DISK_GB}" \
      --os-disk-delete-option Delete --nic-delete-option Delete \
      --public-ip-sku Standard \
      --custom-data "${USERDATA}" \
      --tags "project=SharpeningOrExpansion" "worker=${w}" "config=${CONFIG}" \
      -o none 2>/tmp/soe_create_${w}.err; then
    created+=("${NAME}")
    PRINCIPAL=$(az vm show -g "${RG}" -n "${NAME}" --query identity.principalId -o tsv)
    # Least privilege: the VM needs to read and write blobs and nothing else.
    az role assignment create --assignee-object-id "${PRINCIPAL}" \
      --assignee-principal-type ServicePrincipal \
      --role "Storage Blob Data Contributor" --scope "${SCOPE}" -o none \
      || echo "[launch] WARN: could not grant blob access to ${NAME}; azcopy will fail"
  else
    failed+=("${NAME}")
    echo "[launch] FAILED ${NAME}: $(tail -2 /tmp/soe_create_${w}.err | tr '\n' ' ')"
  fi
  rm -f "${USERDATA}"
done

echo ""
echo "[launch] created ${#created[@]}/${FLEET}: ${created[*]:-none}"
if ((${#failed[@]} > 0)); then
  echo "[launch] failed ${#failed[@]}: ${failed[*]}"
  # Partial capacity is the normal case for GPU spot. The steal pass means a smaller fleet
  # still completes the whole run, just more slowly -- so this is not a reason to abort.
  echo "[launch] NOTE: the run completes with a partial fleet. n_workers in the config stays"
  echo "[launch]       as-is -- it defines the problem partition and must NOT be lowered to"
  echo "[launch]       match the fleet size. Live workers steal the unclaimed shards."
fi
((${#created[@]} > 0)) || { echo "[launch] no VMs created; see README step 1-4 (eligibility, quota, offering, capacity)"; exit 1; }

cat <<MSG

  Bootstrap takes ~10 min (driver check, clone, uv install). Watch one:
    az vm run-command invoke -g ${RG} -n ${created[0]} --command-id RunShellScript \\
      --scripts 'tail -30 /var/log/soe-bootstrap.log /var/log/soe-run.log'

  Heartbeats (a VM with no heartbeat is billing and doing nothing -- reap it):
    azcopy list '${AZ_CONTAINER_URL}' --output-level=essential | grep heartbeat

  TEAR DOWN EVERYTHING (this is the only complete teardown):
    az group delete -n ${RG} --yes --no-wait
MSG
