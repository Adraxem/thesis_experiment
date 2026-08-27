"""
config.py — the deployment configuration space (RQ1).

A "config" is one point in the deployment space whose power waveform we measure.
Everything downstream (sweep -> dataset -> predictor -> optimizer) speaks in these
dataclasses so the feature encoding stays in one place.

Targets the Jetson Orin Nano by default (power modes / precisions set accordingly),
but nothing here is board-specific except DEFAULT_POWER_MODES.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
from itertools import product
from typing import Iterable
import platform


# ---- Enumerations of the deployment levers -------------------------------------

# Precisions we can build with TensorRT. FP8/INT4 only on boards that support them;
# the sweep will skip unsupported ones at runtime.
PRECISIONS = ["FP32", "FP16", "INT8", "FP8", "INT4"]

# Vision + one small LLM (Llama-3.2-1B) for the burst/decode regime.
MODELS = ["resnet18", "resnet50", "mobilenet_v3_large", "yolov8n", "llama3.2-1b"]

BATCH_SIZES = [1, 2, 4, 8, 16]

# Orin Nano power modes. `nvpmodel -q` at runtime is the source of truth; these are
# the labels we sweep. (AGX would add MAXN + 15/30/50W; auto-detected in power/.)
DEFAULT_POWER_MODES = ["7W", "15W", "25W", "MAXN"]

# Scheduling / runtime knobs that also shape the waveform.
SCHEDULES = ["default", "single_stream", "back_to_back"]  # request pacing


@dataclass(frozen=True)
class DeployConfig:
    model: str
    precision: str = "FP16"
    batch_size: int = 1
    power_mode: str = "25W"
    schedule: str = "default"
    mode: str = "inference"          # "inference" or "train" (on-device training/fine-tune)
    # LLM-only knobs (ignored for vision models)
    seq_len: int = 128          # prefill length
    gen_tokens: int = 64        # decode length

    def is_llm(self) -> bool:
        return self.model.startswith("llama") or "llm" in self.model

    def tag(self) -> str:
        base = f"{self.model}_{self.mode}_{self.precision}_b{self.batch_size}_{self.power_mode}_{self.schedule}"
        if self.is_llm():
            base += f"_s{self.seq_len}_g{self.gen_tokens}"
        return base

    # ---- feature encoding used by the predictor (RQ2) --------------------------
    def to_features(self) -> dict:
        """Numeric encoding of a config for the learned predictor."""
        f = {
            "precision_bits": {"FP32": 32, "FP16": 16, "INT8": 8, "FP8": 8, "INT4": 4}[self.precision],
            "is_int": 1 if self.precision in ("INT8", "INT4") else 0,
            "batch_size": self.batch_size,
            "power_budget_w": {"7W": 7, "15W": 15, "25W": 25, "MAXN": 40}[self.power_mode],
            "is_llm": 1 if self.is_llm() else 0,
            "seq_len": self.seq_len if self.is_llm() else 0,
            "gen_tokens": self.gen_tokens if self.is_llm() else 0,
            "sched_back_to_back": 1 if self.schedule == "back_to_back" else 0,
            "is_train": 1 if self.mode == "train" else 0,
        }
        # one-hot the model family
        for m in MODELS:
            f[f"model_{m}"] = 1 if self.model == m else 0
        return f

    def as_row(self) -> dict:
        return asdict(self)


# ---- Sweep construction --------------------------------------------------------

@dataclass
class SweepSpec:
    """Which slice of the space to actually measure. Keep it small first."""
    models: list[str] = field(default_factory=lambda: ["resnet18", "mobilenet_v3_large", "yolov8n", "llama3.2-1b"])
    precisions: list[str] = field(default_factory=lambda: ["FP16", "INT8"])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 4, 8])
    power_modes: list[str] = field(default_factory=lambda: ["15W", "25W", "MAXN"])
    schedules: list[str] = field(default_factory=lambda: ["default"])
    modes: list[str] = field(default_factory=lambda: ["inference"])  # add "train" to capture training power
    llm_seq_lens: list[int] = field(default_factory=lambda: [128])
    llm_gen_tokens: list[int] = field(default_factory=lambda: [64])

    def iter_configs(self) -> Iterable[DeployConfig]:
        for model, prec, bs, pm, sched, mode in product(
            self.models, self.precisions, self.batch_sizes, self.power_modes,
            self.schedules, self.modes
        ):
            if model.startswith("llama") or "llm" in model:
                for sl, gt in product(self.llm_seq_lens, self.llm_gen_tokens):
                    yield DeployConfig(model, prec, bs, pm, sched, mode=mode,
                                       seq_len=sl, gen_tokens=gt)
            else:
                yield DeployConfig(model, prec, bs, pm, sched, mode=mode)


def detect_board() -> str:
    """Best-effort board id; refined by power.telemetry on the real device."""
    if platform.machine() != "aarch64":
        return "dev-x86"  # laptop / CI
    try:
        with open("/proc/device-tree/model") as fh:
            return fh.read().strip("\x00").strip()
    except Exception:
        return "jetson-unknown"


if __name__ == "__main__":
    spec = SweepSpec()
    cfgs = list(spec.iter_configs())
    print(f"board: {detect_board()}")
    print(f"{len(cfgs)} configs in default sweep")
    for c in cfgs[:5]:
        print(" ", c.tag())
