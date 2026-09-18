"""Prompt rendering. The final string is produced here, hashed, and frozen as a fixture.

Rendering deliberately happens in *our* code rather than being left to the engine, so that
``prompt_sha`` is a property we can test offline. The failure this prevents is real: applying
a chat template to a base model, or dropping R1-Distill's ``<think>`` preamble, silently
changes what is being measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from soe.config import ModelSpec
from soe.data.canonical import Problem
from soe.ids import prompt_sha

PROMPT_DIR = Path(__file__).resolve().parents[3] / "configs" / "prompts"
NATIVE_SENTINEL = "__NATIVE_CHAT__"

# Held fixed across runs; part of the prompt hash. Deliberately short and arithmetic-only so
# they prime format rather than content.
FEWSHOT_EXEMPLARS = [
    {"question": "What is $12 \\times 11$?", "solution": "$12 \\times 11 = 132$.", "answer": "132"},
    {"question": "Find the remainder when $2^{10}$ is divided by $7$.",
     "solution": "$2^{10} = 1024$ and $1024 = 146 \\cdot 7 + 2$.", "answer": "2"},
    {"question": "How many positive divisors does $12$ have?",
     "solution": "$12 = 2^2 \\cdot 3$, so it has $(2+1)(1+1) = 6$ divisors.", "answer": "6"},
    {"question": "If $x + 3 = 10$, what is $x^2$?",
     "solution": "$x = 7$, so $x^2 = 49$.", "answer": "49"},
]


@dataclass(frozen=True, slots=True)
class PromptPayload:
    text: str | None                     # completion-mode prompt
    messages: list[dict[str, str]] | None  # chat-mode, for apply_chat_template
    prompt_sha: str
    variant: str


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(PROMPT_DIR), undefined=StrictUndefined, keep_trailing_newline=True
    )


def render(problem: Problem, spec: ModelSpec, variant: str) -> PromptPayload:
    if variant not in spec.prompt_variants:
        raise ValueError(
            f"{spec.key}: variant {variant!r} not permitted; allowed {spec.prompt_variants}"
        )
    tpl = _env().get_template(f"{variant}.jinja")
    rendered = tpl.render(question=problem.question, exemplars=FEWSHOT_EXEMPLARS)

    if rendered.strip() == NATIVE_SENTINEL:
        messages = [
            {
                "role": "user",
                "content": problem.question
                + "\n\nPlease reason step by step, and put your final answer within \\boxed{}.",
            }
        ]
        # The hash covers the logical messages; the tokenizer-applied string is hashed on the
        # node and compared against fixtures/prompts_golden/.
        key = f"native_chat|{spec.key}|{messages[0]['content']}"
        return PromptPayload(None, messages, prompt_sha(key), variant)

    return PromptPayload(rendered, None, prompt_sha(rendered), variant)
