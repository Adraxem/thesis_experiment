"""
models/base.py — a uniform runner interface so the sweep driver treats every model
the same: load once, then run N inference iterations while the PowerLogger records.

Each concrete runner tries the real backend (torch / TensorRT / transformers) and
falls back to a timing-only stub if the dependency or hardware is absent, so the
whole pipeline runs on your laptop today.
"""
from __future__ import annotations
import abc
import importlib
import time
from dataclasses import dataclass


def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


@dataclass
class RunResult:
    n_inferences: int
    wall_s: float
    backend: str
    extra: dict


class BaseRunner(abc.ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = "stub"

    @abc.abstractmethod
    def load(self) -> None: ...

    @abc.abstractmethod
    def run(self, n_iters: int) -> RunResult: ...

    def train(self, n_iters: int) -> RunResult:
        """On-device training/fine-tune for n_iters steps (RQ1, training regime).
        Default is a timing stub; VisionRunner/LLMRunner override with real backprop."""
        return self._stub_run(n_iters, 0.05)

    # helper: sleep-based stub so timings are non-zero off-device
    def _stub_run(self, n_iters: int, per_iter_s: float) -> RunResult:
        t0 = time.perf_counter()
        for _ in range(n_iters):
            time.sleep(per_iter_s)
        return RunResult(n_iters, time.perf_counter() - t0, "stub",
                         {"note": "no backend/hardware; timing stub"})


def make_runner(cfg):
    """Factory: pick the vision or LLM runner for a DeployConfig."""
    from .vision import VisionRunner
    from .llm import LLMRunner
    if cfg.is_llm():
        return LLMRunner(cfg)
    return VisionRunner(cfg)
