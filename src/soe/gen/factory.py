"""Backend selection. The ONLY module that may import the vLLM backend."""

from __future__ import annotations

from soe.gen.backend import GenerationBackend


def make_backend(name: str, **kwargs) -> GenerationBackend:
    if name == "mock":
        from soe.gen.mock_backend import MockBackend

        return MockBackend(**kwargs)
    if name == "replay":
        from soe.gen.replay_backend import ReplayBackend

        return ReplayBackend(**kwargs)
    if name == "vllm":
        # Imported inside the function so that `import soe.cli` never pulls in torch.
        from soe.gen.vllm_backend import VLLMBackend

        return VLLMBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; expected one of mock, replay, vllm")
