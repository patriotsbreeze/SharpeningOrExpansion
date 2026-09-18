# Azure runbook

Copy-paste order. **Steps 0–4 are gates, not formalities** — each fails differently, and
step 3 has a lead time measured in days.

> **The budget is not the constraint here; capacity is.** Even the worst realistic case —
> paying full pay-as-you-go with zero spot capacity — lands inside $600. So pre-authorise
> yourself to fall back to on-demand rather than losing days waiting for spot that never
> arrives. See *Sizing*.

## 0. Subscription eligibility — a hard gate

```bash
az account show --query "{sub:id, name:name, state:state}" -o table
SUB=$(az account show --query id -o tsv)
```

A **Free Trial subscription cannot raise GPU quota and is excluded from Spot entirely.** You
need Pay-as-you-go, EA, Sponsored, or CSP. Nothing below works otherwise.

## 1. Discover the real quota family names — never hardcode them

Portal display names and ARM names are not a mechanical transform (portal
`Standard NCv3 Family vCPUs` is ARM `standardNCSv3Family` — the `S` moves). Ask the API:

```bash
az vm list-skus -l "$LOCATION" --size Standard_NC40ads_H100_v5 --query "[0].family" -o tsv
az vm list-skus -l "$LOCATION" --size Standard_NC24ads_A100_v4 --query "[0].family" -o tsv

for R in eastus eastus2 southcentralus westus3 swedencentral westeurope; do
  echo "### $R"
  az vm list-usage -l "$R" -o table | grep -iE 'H100|A100|Total Regional|spot|low-priority'
done
```

Grep for **both** `Spot` and `Low-priority` — the CLI still prints the legacy label.

## 2. Confirm the SKU is even offered to your subscription

```bash
az vm list-skus -l "$LOCATION" --size Standard_NC40ads_H100_v5 --all -o table
```

`--all` is mandatory: without it, `NotAvailableForSubscription` rows are **silently hidden**.
This is a third failure mode, distinct from quota and from capacity — a SKU can be
quota-approved, have capacity, and still not be offered to you.

## 3. Request quota — two separate requests

```bash
az extension add --name quota
SCOPE="/subscriptions/$SUB/providers/Microsoft.Compute/locations/$LOCATION"
az quota list --scope "$SCOPE" -o table          # cross-check names against step 1

# On-demand pool
az quota update --resource-name <ARM_NAME_FROM_STEP_1> --scope "$SCOPE" \
  --limit-object value=80 --resource-type dedicated

# Spot pool -- a SEPARATE pool with its OWN resource name, not a resource-type on the
# dedicated family. Take the exact name from the spot/low-priority row that step 1 printed
# (commonly `lowPriorityCores`, but read it, do not assume):
az quota update --resource-name <SPOT_ROW_NAME_FROM_STEP_1> --scope "$SCOPE" \
  --limit-object value=80

az quota request status list --scope "$SCOPE" -o table
```

⚠️ `--resource-type lowPriority` is **not** a Compute discriminator — passing it against the
dedicated family name silently requests the wrong pool, and you discover that when every spot
create fails allocation. The spot quota is its own row with its own name.

40 vCPU per NC40ads worker, 24 per NC24ads — so 80 buys two concurrent H100 workers.
Requesting is free: **ask in two or three regions in parallel.** If `az quota` misbehaves, the
portal path (Quotas → Compute → My quotas → search "spot" → New Quota Request) is safer.

## 4. Pick the region on eviction rate, not price

```bash
az extension add --name resource-graph

az graph query -q "SpotResources
  | where type =~ 'microsoft.compute/skuspotevictionrate/location'
  | where sku.name in~ ('standard_nc40ads_h100_v5','standard_nc24ads_a100_v4')
  | project skuName=tostring(sku.name), location, rate=tostring(properties.evictionRate)
  | order by skuName asc, location asc" -o table
```

The same `SpotResources` table also exposes `skuspotpricehistory`, which is **the only
authoritative spot-price source reachable without the retail API** — use it to replace the
unverified figures below with real numbers for your region.

## 5. Storage — in its own resource group

```bash
az group create -n soe-data -l "$LOCATION"
az storage account create -g soe-data -n <STORAGE_ACCOUNT> -l "$LOCATION" \
  --sku Standard_LRS --kind StorageV2
# --auth-mode login uses the DATA plane, which subscription Owner alone does not grant.
# Give yourself the data role first (propagation takes a minute or two):
az role assignment create --assignee "$(az ad signed-in-user show --query id -o tsv)" \
  --role "Storage Blob Data Contributor" \
  --scope "$(az storage account show -n <STORAGE_ACCOUNT> --query id -o tsv)"
az storage container create --account-name <STORAGE_ACCOUNT> -n soe --auth-mode login
export AZ_CONTAINER_URL="https://<STORAGE_ACCOUNT>.blob.core.windows.net/soe"
```

Deliberately a **different** resource group from the compute, so tearing down the fleet cannot
delete your results.

## 6. Launch the fleet

```bash
export LOCATION=eastus2 STORAGE_ACCOUNT=<acct> FLEET=4
export CONFIG=configs/experiments/stage1_tierA.yaml
./azure/launch.sh
```

One single-GPU VM per worker, each with a system-assigned identity scoped to
`Storage Blob Data Contributor` on that storage account only.

Pay-as-you-go fallback, when spot capacity does not materialise:

```bash
PRIORITY=Regular ./azure/launch.sh      # spot-only flags are omitted automatically
```

While it runs, reap anything billing without working — cloud-init failure does **not** surface
as a provisioning failure, so the portal will show a dead VM as `Running`:

```bash
RG=soe-run ./azure/reap.sh              # report
RG=soe-run DRY_RUN=0 ./azure/reap.sh    # delete
```

**Partial capacity is normal and fine.** If you ask for 4 and get 2, the run still completes —
live workers steal the unclaimed shards. **Do not lower `n_workers` in the config to match the
fleet**: it defines the problem partition, and changing it against an existing artifact tree
reassigns problems to different shards while leaving seeds unchanged, writing the same sample
under two paths. `soe verify` reports that as duplicate seeds.

## 7. Preflight on real weights — before the full matrix

```bash
ssh azureuser@<ip>
cd /opt/soe && source /etc/profile.d/soe.sh

cat /opt/azurehpc/component_versions.txt   # which CUDA/driver this image actually shipped
nvidia-smi
soe doctor --registry                      # resolves every HF id. Costs nothing.

SOE_WORKER=0 scripts/launch_node.sh configs/experiments/pilot_gpu.yaml
soe verify configs/experiments/pilot_gpu.yaml --root "$SOE_ROOT"
```

Read one shard by eye before going further: the prompt rendered as intended, `finish_reason`
not universally `length`, and `max_tokens_effective` what you expect.

## 8. Analysis — anywhere, no GPU

Pull the **complete** tree first. Each VM only ever held its own shards plus everyone's
markers, so the deep verify — and the cross-machine checks it performs: global seed
uniqueness, equal *n* per problem, one `problem_uid` per problem index — can only run here,
once, against everything.

```bash
mkdir -p runs && cd runs
azcopy login                                 # or AZCOPY_AUTO_LOGIN_TYPE=AZCLI after `az login`

# If AZ_CONTAINER_URL carries a SAS, the wildcard must go BEFORE the "?" -- appending it after
# folds the path into the signed query and the request targets the wrong prefix. With a plain
# https URL, append it normally.
SRC="${AZ_CONTAINER_URL%%\?*}/*"
[[ "$AZ_CONTAINER_URL" == *"?"* ]] && SRC="$SRC?${AZ_CONTAINER_URL#*\?}"
azcopy copy "$SRC" . --recursive             # shards AND markers this time

# Grade here too. A VM that was evicted mid-run never reached its own grading step, so
# its shards arrive ungraded -- and grading is CPU-only and idempotent via its own markers,
# so re-running it over the whole tree is cheap and safe.
soe grade   configs/experiments/stage1_tierA.yaml --root . --graders fastint,mathverify

soe verify  configs/experiments/stage1_tierA.yaml --root . # deep by default; must pass
soe figures configs/experiments/stage1_tierA.yaml --root . \
    --base qwen25m7b_base --rl qwen25m7b_simplerl0
```

The per-VM verify inside `launch_node.sh` deliberately runs `--no-deep`: on a resumed VM most
shards are not local, and a deep check would report a failure for every absent one.

## 9. Tear down

```bash
./azure/teardown.sh
```

---

## Sizing

Prefer the **H100 NC line**: Azure states it is *"only deploying net new capacity for the
latest generation of the NC product line, the NCads_H100_v5-series"* — the A100 pool is frozen
and will only get harder to allocate.

| SKU | GPU | Hours needed | Est. total |
|---|---|---|---|
| **NC40ads_H100_v5** spot | 1× H100 NVL **94GB** | 55–67 | **$71–167** |
| NC24ads_A100_v4 spot | 1× A100 80GB PCIe + 960GB NVMe | 130–145 | $88–99 |
| **NC40ads_H100_v5 PAYG** | — | 55–67 | **$384–468** ← safe fallback |
| NC24ads_A100_v4 PAYG | — | 130–145 | $477–533 |
| ND96isr_H100_v5 PAYG | 8× H100 80GB | ~7–8 node-h | $688–786 ❌ |

Hour counts are throughput-adjusted, not a flat 120: the A100 80GB **PCIe** part is slower
than the A100 SXM the original estimate assumed, and H100 is roughly 1.8–2.2× faster, needing
about half the hours. That inversion is why **PAYG H100 is cheaper than PAYG A100 here**.

Avoid 8-GPU ND nodes: they bust the budget at PAYG, and one eviction takes all eight workers.

> ⚠️ **Every price above is unverified.** `prices.azure.com`, `learn.microsoft.com` and every
> aggregator were unreachable from the authoring environment, and the aggregators that *were*
> indexed disagree by 2× on the NC40ads spot rate ($1.29 vs $2.49) — one of them also reports
> its GPU as 80GB when it is 94GB. The *ranking* is robust; the absolute totals are not.
> Re-derive them from the `skuspotpricehistory` query in step 4 before committing spend.
> The conclusion "PAYG fits the budget" holds even at 1.5× the quoted rates.

Also unverified: the 1.8–2.2× H100-over-A100 throughput ratio is bandwidth-derived, not
benchmarked, and every hour count above depends on it. Measure it in step 7.

## Dangerous defaults

Each is silent — no error, just data loss or money burned.

| Default | What it does | Mitigation |
|---|---|---|
| **`az storage blob sync` forces `--delete-destination=true`** | A worker pushing its own partial subtree **deletes every other worker's shards.** Not mentioned in `--help`. | Never use it. `scripts/objstore.sh` uses `azcopy copy`, which has no prune semantics; a test asserts neither `az storage blob sync` nor `azcopy sync` appears in the scripts. |
| `azcopy copy` defaults `--recursive` **false** | Uploads only top-level files and **exits 0**. (`azcopy sync` defaults it true — the inconsistency is the trap.) | Always explicit; a test enforces it on every tree copy. |
| `--eviction-policy` defaults to **Deallocate** | Evicted VMs keep billing for disks **and keep consuming spot quota**, so replacements cannot allocate — the fleet strangles itself. | `--eviction-policy Delete` in `launch.sh`. |
| Local NVMe is wiped on redeploy and returns **RAW** | A first-boot-only `mkfs` leaves later instances with no scratch mount, failing deep inside the run. The image ships no `nvme-raid` unit. | `bootstrap.sh` does RAID+mkfs idempotently on **every** boot. |
| **cloud-init failure is not provisioning failure** | The portal shows the VM `Running` while bootstrap died at step 2. A broken GPU VM bills at full rate for hours. | `bootstrap.sh` publishes a per-stage heartbeat to blob. A VM with no heartbeat is dead — reap it. |
| Scheduled Events is **lazily enabled** | First call takes ~2 min, events flow after ~5, and it self-disables after 24h without polling. A watcher started with the job is blind exactly when an early eviction costs most. | A systemd poller starts at **boot** and writes a sentinel; `launch_node.sh` only reacts to it. It only ever GETs — POST-acknowledging a Preempt event forfeits the remaining notice window. |
| Guest-side `shutdown -h now` leaves the VM **`Stopped`**, not `Stopped (deallocated)` | **You keep paying for compute.** A guest-side timer is not a cost control. | Tear down the resource group, or self-delete via managed identity. |
| `az vmss create` now defaults to **Flexible** | Instance names become `{vmss}_{guid}` — no numeric index, so no stable worker identity. | Use the `az vm create` loop in `launch.sh`. Azure declines to guarantee Uniform IDs are `0..N-1` anyway, and reuses them after deletion. |
| `az vm delete` does not cascade | Orphaned public IPs and disks bill indefinitely. | Dedicated resource group + `teardown.sh`. |

## Verify before relying on it

**No `az` command here was executed against a live subscription.** Specifically:

- `az quota update` syntax — from a Microsoft skills repo carrying its own staleness warning.
- The exact ARM family name for `NCads_A100_v4`, and the verbatim label of the per-family
  *spot* quota row. Step 1 prints the truth; don't paste a guess.
- Whether `az vm auto-shutdown` yields `Stopped` or `Stopped (deallocated)`. If the former it
  is not a cost control at all. Check `az vm get-instance-view --query instanceView.statuses`.
- `microsoft-dsvm:ubuntu-hpc:2404:latest` — that image family's README still lists `2204`, so
  22.04 is not withdrawn; 24.04 being current rests on release cadence.
- Whether NVIDIA Fabric Manager (shipped in the image, targets NVSwitch) fails noisily at boot
  on a single-GPU NC VM. Check `journalctl` on the first node.
- Whether a CUDA 12.x vLLM wheel runs on the image's driver. Step 7 smoke-tests this.
- `az compute-recommender spot-placement-recommender` does **not** exist as a CLI extension —
  if you see it suggested elsewhere it will fail. Use the `resource-graph` queries instead.
- Regional capacity: SKU availability is subscription-scoped, so **any published region list is
  wrong for you by construction.** Only `az vm list-skus -l <L> --all` with empty
  `Restrictions` is authoritative. The regions named here are candidates to test.
