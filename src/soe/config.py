"""Pydantic schemas for the registries and experiment configs.

Everything is ``extra="forbid"`` and frozen: a typo in a YAML key should fail at load, not
silently do nothing to a 120-GPU-hour run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["base", "rl_zero", "distill", "rl_ckpt"]
Tier = Literal["A", "B", "control_contaminated", "anchor"]
AnswerType = Literal["integer_0_999", "integer", "expr"]
Variant = Literal["fewshot4_math", "zeroshot_boxed", "native_chat"]


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(Frozen):
    key: str
    seed_idx: int = Field(ge=0, lt=64)
    hf_id: str
    revision: str
    family: str
    role: Role
    base_of: str | None = None
    rl_step: int | None = None
    native_context_len: int = Field(gt=0)
    context_len: int = Field(gt=0)
    rope_scaling: dict | None = None
    release_date: date
    prompt_variants: list[Variant]
    stop: list[str] = Field(default_factory=list)
    dtype: Literal["bfloat16"] = "bfloat16"

    @property
    def is_mock(self) -> bool:
        return self.hf_id.startswith("mock://")

    @model_validator(mode="after")
    def _check(self) -> ModelSpec:
        if self.context_len > self.native_context_len and self.rope_scaling is None:
            raise ValueError(
                f"{self.key}: context_len {self.context_len} exceeds native "
                f"{self.native_context_len} without rope_scaling. Extrapolating past the "
                f"trained window silently degrades quality; set rope_scaling explicitly or "
                f"lower context_len."
            )
        if self.role in ("rl_zero", "distill", "rl_ckpt") and not self.base_of:
            raise ValueError(f"{self.key}: role={self.role} requires base_of")
        if self.role == "rl_ckpt" and self.rl_step is None:
            raise ValueError(f"{self.key}: role=rl_ckpt requires rl_step")
        if not self.prompt_variants:
            raise ValueError(f"{self.key}: at least one prompt variant required")
        return self


class DatasetSpec(Frozen):
    key: str
    seed_idx: int = Field(ge=0, lt=64)
    hf_path: str
    hf_config: str | None = None
    split: str
    revision: str
    tier: Tier
    release_date: date
    answer_type: AnswerType
    expected_rows: int | None = None

    @property
    def is_mock(self) -> bool:
        return self.hf_path.startswith("mock://")


class SamplingSpec(Frozen):
    n_total: int = Field(gt=0, le=1 << 20)
    chunk_size: int = Field(gt=0)
    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0, default=1.0)
    top_k: int = -1
    max_new_tokens: int = Field(gt=0)
    max_prompt_tokens: int = Field(gt=0, default=1024)
    seed_epoch: int = Field(ge=0, lt=64, default=0)

    @model_validator(mode="after")
    def _check(self) -> SamplingSpec:
        if self.chunk_size > self.n_total:
            raise ValueError(f"chunk_size {self.chunk_size} exceeds n_total {self.n_total}")
        if self.n_total % self.chunk_size:
            raise ValueError(
                f"n_total {self.n_total} must be a multiple of chunk_size {self.chunk_size} "
                f"so chunk boundaries are deterministic"
            )
        return self

    @property
    def sampling_id(self) -> str:
        """Stable hash of the sampling distribution.

        Deliberately EXCLUDES ``chunk_size``: chunking is a checkpointing concern, and the seed
        of sample j depends only on j. That means chunk_size may be changed mid-run after a
        preemption without invalidating any existing shard.
        """
        payload = {
            "n_total": self.n_total,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "seed_epoch": self.seed_epoch,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:8]


class ArmSpec(Frozen):
    """One (model, dataset, variant) cell of the experiment matrix."""

    model_key: str
    dataset_key: str
    variant: Variant
    sampling_override: SamplingSpec | None = None


class ExperimentConfig(Frozen):
    exp_id: str
    description: str = ""
    backend: Literal["vllm", "mock", "replay"] = "vllm"
    sampling: SamplingSpec
    arms: list[ArmSpec]
    n_workers: int = Field(gt=0, default=8)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        seen = {(a.model_key, a.dataset_key, a.variant) for a in self.arms}
        if len(seen) != len(self.arms):
            raise ValueError("duplicate (model, dataset, variant) arm in experiment")
        return self

    def sampling_for(self, arm: ArmSpec) -> SamplingSpec:
        return arm.sampling_override or self.sampling
