"""vLLM backend. Imported only via ``factory.make_backend`` so nothing else pulls in torch.

**We never use vLLM's ``n > 1``.** Each sample is its own ``n=1`` request carrying an explicit
seed. The child-seed derivation for ``n > 1`` (``seed=S`` giving children ``S..S+n-1``) is an
implementation detail that has changed shape across vLLM releases, and betting 120 GPU-hours
on it is a bad trade when prefix caching already makes the repeated prompt nearly free. The
payoff is that the sample-to-seed map becomes *recorded data* rather than assumed semantics,
so ``soe verify`` can prove seed uniqueness from the artifacts themselves.

Topology: ``tensor_parallel_size=1`` with one engine per GPU and problems sharded across
workers. A 7B in bf16 is ~15 GB, so eight independent engines beat tensor parallelism on
throughput at this scale.
"""

from __future__ import annotations

from collections.abc import Sequence

from soe.config import ModelSpec
from soe.gen.backend import GenRequest, GenSample


class VLLMBackend:
    def __init__(
        self,
        *,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int | None = None,
        enable_prefix_caching: bool = True,
        enforce_eager: bool = False,
        download_dir: str | None = None,
    ) -> None:
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.enable_prefix_caching = enable_prefix_caching
        self.enforce_eager = enforce_eager
        self.download_dir = download_dir
        self._llm = None
        self._tok = None
        self._spec: ModelSpec | None = None

    def load(self, spec: ModelSpec) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self._spec = spec
        kwargs = dict(
            model=spec.hf_id,
            revision=spec.revision,
            tokenizer_revision=spec.revision,
            dtype=spec.dtype,
            max_model_len=spec.context_len,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_prefix_caching=self.enable_prefix_caching,
            enforce_eager=self.enforce_eager,
            trust_remote_code=True,
            seed=0,
        )
        if spec.rope_scaling:
            # Only ever set when context_len exceeds the trained window; config.py enforces
            # that pairing, so an accidental extrapolation cannot reach here silently.
            kwargs["hf_overrides"] = {"rope_scaling": spec.rope_scaling}
        if self.max_num_seqs:
            kwargs["max_num_seqs"] = self.max_num_seqs
        if self.download_dir:
            kwargs["download_dir"] = self.download_dir

        self._llm = LLM(**kwargs)
        self._tok = AutoTokenizer.from_pretrained(
            spec.hf_id, revision=spec.revision, trust_remote_code=True
        )

    @property
    def fingerprint(self) -> str:
        import torch
        import vllm

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return f"vllm{vllm.__version__}|torch{torch.__version__}|{gpu}"

    def _render(self, payload: str | list[dict]) -> str:
        if isinstance(payload, str):
            return payload
        return self._tok.apply_chat_template(
            payload, tokenize=False, add_generation_prompt=True
        )

    def count_prompt_tokens(self, payloads: Sequence[str | list[dict]]) -> list[int]:
        texts = [self._render(p) for p in payloads]
        return [len(self._tok(t, add_special_tokens=False).input_ids) for t in texts]

    def generate(self, reqs: Sequence[GenRequest]) -> list[GenSample]:
        from vllm import SamplingParams

        prompts, params = [], []
        for r in reqs:
            prompts.append(self._render(r.prompt))
            params.append(
                SamplingParams(
                    n=1,                       # never n>1; see module docstring
                    seed=r.seed,
                    temperature=r.temperature,
                    top_p=r.top_p,
                    top_k=r.top_k,
                    max_tokens=r.max_tokens,
                    stop=list(r.stop) or None,
                )
            )
        outs = self._llm.generate(prompts, params)

        samples = []
        for r, o in zip(reqs, outs, strict=True):
            c = o.outputs[0]
            samples.append(
                GenSample(
                    request_id=r.request_id,
                    text=c.text,
                    n_prompt_tokens=len(o.prompt_token_ids),
                    n_completion_tokens=len(c.token_ids),
                    finish_reason=c.finish_reason or "stop",
                    stop_str=getattr(c, "stop_reason", None) if isinstance(
                        getattr(c, "stop_reason", None), str
                    ) else None,
                )
            )
        return samples

    def close(self) -> None:
        self._llm = None
        self._tok = None
        self._spec = None
