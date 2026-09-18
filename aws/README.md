# AWS runbook

Copy-paste order. **Step 0 has a lead time measured in hours — do it first.**

## 0. Quota (do this before anything else)

`p5.48xlarge` needs *Running On-Demand P instances* **vCPU** quota (192 vCPUs for one node),
and Spot needs the separate *All P Spot Instance Requests* quota. Both default to **0** on new
accounts and approval is not instant.

```
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-417A185B --desired-value 192   # On-Demand P
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-3819A6DF --desired-value 192   # Spot P
```

## 1. Bucket and role

```
aws s3 mb s3://YOUR-BUCKET
```

Create an instance profile `soe-node` with `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`
on that bucket. Nothing else — the node needs no other AWS permission.

## 2. Pick an AMI

```
aws ssm get-parameters --names \
  /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query 'Parameters[0].Value' --output text
```

## 3. Launch

```
export AMI_ID=ami-...  KEY_NAME=...  SUBNET_ID=subnet-...  SG_ID=sg-...
export S3_URI=s3://YOUR-BUCKET/soe
./aws/launch.sh                       # spot, with automatic on-demand fallback
```

## 4. Preflight — **before** the full matrix

```
ssh ubuntu@<ip>
cd /opt/soe && source /etc/profile.d/soe.sh

soe doctor --registry        # resolves every HF id. Catches a typo for $0 instead of at hour 3.
scripts/launch_node.sh configs/experiments/pilot_gpu.yaml 1     # ~$2
soe verify configs/experiments/pilot_gpu.yaml --root $SOE_ROOT
```

Read `runs/exp=pilot_gpu/gen/.../chunk=0000.jsonl.zst` by eye before going further. Check:
the prompt rendered as intended, `finish_reason` is not universally `length`, and
`max_tokens_effective` is what you expect (see the 4096 note below).

## 5. Full run

```
S3_URI=$S3_URI scripts/launch_node.sh configs/experiments/stage1_tierA.yaml
S3_URI=$S3_URI scripts/launch_node.sh configs/experiments/stage2_olmo_distill.yaml
S3_URI=$S3_URI scripts/launch_node.sh configs/experiments/stage3_rl_trajectory.yaml
```

Resumable. After a spot reclaim, relaunch and re-run the same command: markers are synced
from S3 and completed chunks are skipped.

## 6. Analysis (runs anywhere, no GPU)

```
soe verify  configs/experiments/stage1_tierA.yaml --root $SOE_ROOT
soe figures configs/experiments/stage1_tierA.yaml --root $SOE_ROOT \
    --base qwen25m7b_base --rl qwen25m7b_simplerl0
```

## 7. Terminate

```
aws ec2 terminate-instances --region $REGION --instance-ids i-...
```

An idle `p5.48xlarge` costs about **$55/hour**. Set a billing alarm.

---

## Instance selection

| SKU | GPUs | On-demand | Spot | Interruption |
|---|---|---|---|---|
| **p5.48xlarge** | 8x **H100 80GB** | ~$55/hr | ~$31/hr | **<5%** |
| p4de.24xlarge | 8x A100 **80GB** | ~$27/hr | — | capacity-constrained |
| p4d.24xlarge | 8x A100 **40GB** | ~$22-33/hr | cheaper | 15-20% |
| g6e.48xlarge | 8x **L40S 48GB** | cheaper | good availability | low |

Two corrections to assumptions that are easy to get wrong: **`p4d` is A100 40GB, not 80GB**
(`p4de` is the 80GB SKU), and **`g6e` is L40S, not A100/H100**. 48GB is still ample for a 7B
with a large KV cache, so `g6e` is the sane fallback when H100 capacity is unavailable.

Pricing is aggregator-sourced and region-dependent — **confirm in the console**. Budget:
~100 GPU-hours at p5 spot ≈ 12.5 wall-hours ≈ **$385**.

## Operational notes

- **The 4096 trap.** `Qwen/Qwen2.5-Math-7B` has `max_position_embeddings=4096` and
  `rope_theta=10000` — and 4096 is the *total* budget. A 4-shot prompt is ~900 tokens, so
  `max_new_tokens` must be ~3000. The runner computes and records
  `max_tokens_effective = min(max_new_tokens, context_len - n_prompt_tokens)` per sample; if
  the engine were left to clamp silently, the C5/C6 token accounting would be wrong on the
  single most important arm in the paper.
- **Stagger worker startup.** 8 x ~15 GB of weights loading at once spikes host RAM to ~120 GB
  before the copy to device. `STAGGER=30` (seconds) is the default; raise it if the OOM killer
  appears in `dmesg`.
- **`max_num_seqs`.** Start around 128-256 at 4k context, 32-64 at 32k. Oversubscribing does
  not OOM — vLLM preempts and recomputes, which silently halves throughput.
- **OLMo 3 throughput.** vLLM 0.29.0 serves Olmo3 through the Transformers modeling backend
  rather than a native kernel path. Benchmark it during the pilot before committing a k=256
  budget to Stage 2; if it is badly slow, cut Stage 3 first.
- **Weights on NVMe, not EBS.** `bootstrap.sh` puts `HF_HOME` on the instance-store NVMe.
  Six checkpoints on the root EBS volume is a silent half-hour tax on every boot.
- **S3 sync is markers-first.** Resume pulls only `*.done.json` (kilobytes), never the shards.
