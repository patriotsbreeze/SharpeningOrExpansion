"""Process pool with hard per-task timeouts.

math-verify's internal timeouts are ``signal.alarm``-based, which only works on the main
thread -- so a thread pool cannot protect us. And ``concurrent.futures.ProcessPoolExecutor``
cannot cancel a *running* future, so a completion that sends ANTLR into catastrophic
backtracking hangs a worker forever. ``pebble`` can actually kill the worker, which is why it
is a hard dependency here rather than a preference.

Timed-out samples are recorded as ``grade_status="timeout"``, counted as incorrect, and
quarantined. The quarantine rate is reported in the paper: a base model with a 3% grader
timeout rate is itself a finding, not a nuisance.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

TAIL_BYTES = 4096   # answers live at the end; the megabyte prefix only feeds the parser
OUTER_TIMEOUT_S = 10.0
BATCH_SIZE = 500    # per-sample tasks at ~1ms of work are 90% pickling overhead


@dataclass(frozen=True, slots=True)
class GradeOutcome:
    index: int
    values: dict
    status: str      # "ok" | "timeout" | "error"


def clip_tail(text: str, n_bytes: int = TAIL_BYTES) -> str:
    return text if len(text) <= n_bytes else text[-n_bytes:]


def map_with_timeout(
    fn: Callable[[Sequence], list],
    batches: Sequence[Sequence],
    *,
    workers: int,
    timeout_s: float = OUTER_TIMEOUT_S,
    max_tasks: int = 2000,
) -> list:
    """Run ``fn`` over batches, killing any worker that exceeds ``timeout_s``.

    Falls back to serial execution when pebble is unavailable so that the CPU smoke path and
    CI stay dependency-light.
    """
    try:
        from pebble import ProcessPool
    except ImportError:
        return [fn(b) for b in batches]

    results: list = [None] * len(batches)
    # max_tasks recycles workers so SymPy's memory growth is bounded over millions of calls.
    with ProcessPool(max_workers=workers, max_tasks=max_tasks) as pool:
        futures = [
            (i, pool.schedule(fn, args=(b,), timeout=timeout_s * max(1, len(b))))
            for i, b in enumerate(batches)
        ]
        for i, fut in futures:
            try:
                results[i] = fut.result()
            except Exception:
                results[i] = None  # caller marks the whole batch as timed out
    return results
