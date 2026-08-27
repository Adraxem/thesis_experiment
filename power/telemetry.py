"""
power/telemetry.py — high-rate power/thermal sampling on the Jetson Orin (RQ1).

Reads the on-board INA3221 rails via sysfs (fast, ~ms) and, in parallel, parses
`tegrastats` for the GPU/SOC power fields and temperatures. On a non-Jetson host
(your laptop / CI) it produces a *synthetic* waveform so the whole pipeline runs
end-to-end before you have the board.

Sampling note (matches the thesis scope): tegrastats/INA3221 sample at ~ms, so we
capture PEAK power and THERMAL transients, not µs-scale di/dt. That caveat is
intentional and documented in the proposal.

Usage:
    with PowerLogger(period_ms=5) as log:
        run_inference(...)
    trace = log.trace()          # -> PowerTrace(t, p_total, rails..., temp)
"""
from __future__ import annotations
import glob
import os
import platform
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

import numpy as np

IS_JETSON = platform.machine() == "aarch64"

# INA3221 sysfs hwmon paths differ across JetPack versions; we glob for them.
_HWMON_GLOBS = [
    "/sys/bus/i2c/drivers/ina3221x/*/hwmon/hwmon*/",
    "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*/",
    "/sys/class/hwmon/hwmon*/",
]


@dataclass
class PowerTrace:
    t: np.ndarray                 # seconds, relative to start
    p_total: np.ndarray           # milliwatts, total module power
    rails: dict = field(default_factory=dict)   # rail_name -> mW array
    temp: np.ndarray | None = None              # deg C (max zone), optional
    period_ms: float = 5.0
    source: str = "mock"

    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0


# ---------------------------------------------------------------------------
# Real-device rail discovery
# ---------------------------------------------------------------------------
def _discover_rails() -> list[tuple[str, str]]:
    """Return [(label, curr_input_path, ...)] -> we read power = in*curr."""
    found = []
    for pattern in _HWMON_GLOBS:
        for base in glob.glob(pattern):
            # newer kernels expose 'curr{n}_input' (mA) and 'in{n}_input' (mV)
            labels = sorted(glob.glob(os.path.join(base, "*_label")))
            for lab in labels:
                try:
                    name = open(lab).read().strip()
                except Exception:
                    continue
                idx = re.search(r"(\d+)", os.path.basename(lab))
                if not idx:
                    continue
                n = idx.group(1)
                curr = os.path.join(base, f"curr{n}_input")
                volt = os.path.join(base, f"in{n}_input")
                if os.path.exists(curr) and os.path.exists(volt):
                    found.append((name, curr, volt))
    return found


def _read_int(path: str) -> float:
    try:
        return float(open(path).read().strip())
    except Exception:
        return float("nan")


def _read_max_temp() -> float:
    best = float("nan")
    for zone in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        v = _read_int(zone)
        if v == v:  # not nan
            best = max(best if best == best else -1e9, v / 1000.0)
    return best


# ---------------------------------------------------------------------------
# Synthetic waveform (laptop / CI)
# ---------------------------------------------------------------------------
def _synth_trace(duration_s: float, period_ms: float, cfg=None) -> PowerTrace:
    """A plausible power waveform: idle floor + inference bursts, with an LLM
    decode ripple and a slow thermal ramp. Deterministic-ish per config tag."""
    n = max(4, int(duration_s * 1000.0 / period_ms))
    t = np.arange(n) * (period_ms / 1000.0)
    rng = np.random.default_rng(abs(hash(getattr(cfg, "tag", lambda: "x")())) % (2**32) if cfg else 0)

    is_llm = bool(cfg and cfg.is_llm())
    bits = 16
    budget = 25.0
    if cfg:
        bits = cfg.to_features()["precision_bits"]
        budget = cfg.to_features()["power_budget_w"]
        bs = cfg.batch_size
    else:
        bs = 1

    floor = 2500.0 + 200 * np.log2(max(bs, 1))                 # mW idle-ish
    peak = 1000.0 * budget * (0.55 + 0.45 * (bits / 32.0)) * (0.8 + 0.15 * np.log2(max(bs, 1) + 1))
    p = np.full(n, floor)

    is_train = bool(cfg and getattr(cfg, "mode", "inference") == "train")
    if is_train:
        # TRAINING regime: high sustained draw (fwd+bwd+opt) with periodic "sync"
        # dips where a real cluster would stall on gradient all-reduce. This per-device
        # pattern is the building block Part ii superposes across many GPUs.
        steps = 10
        phase = np.linspace(0, steps * 2 * np.pi, n)
        compute = 0.82 + 0.10 * (np.sin(phase) > -0.7)     # mostly high
        sync_dip = 0.35 * (np.abs(np.sin(phase / 2.0)) > 0.985)  # brief all-reduce stalls
        level = np.clip(compute - sync_dip, 0.25, 1.0)
        p = floor + (peak - floor) * level
    elif is_llm:
        # prefill spike then decode ripple
        p[: n // 6] = peak * (0.9 + 0.1 * rng.random(n // 6))
        ripple = 0.6 + 0.25 * np.sin(np.linspace(0, 20 * np.pi, n - n // 6))
        p[n // 6:] = floor + (peak - floor) * ripple
    else:
        # periodic inference bursts
        burst = (np.sin(np.linspace(0, 12 * np.pi, n)) > 0.2).astype(float)
        p = floor + (peak - floor) * burst

    p += rng.normal(0, peak * 0.03, n)              # measurement noise
    p = np.clip(p, floor * 0.8, None)

    # slow thermal ramp toward a plateau
    temp0, tmax = 38.0, 38.0 + (0.55 if is_train else 0.35) * budget  # training ramps hotter
    temp = tmax - (tmax - temp0) * np.exp(-t / max(duration_s / 2, 1e-3))
    temp += rng.normal(0, 0.2, n)

    return PowerTrace(t=t, p_total=p, rails={"VDD_GPU_SOC": p * 0.6, "VDD_CPU_CV": p * 0.25},
                      temp=temp, period_ms=period_ms, source="mock")


# ---------------------------------------------------------------------------
# The logger
# ---------------------------------------------------------------------------
class PowerLogger:
    def __init__(self, period_ms: float = 5.0, cfg=None, use_tegrastats: bool = False):
        self.period_ms = period_ms
        self.cfg = cfg
        self.use_tegrastats = use_tegrastats
        self._rails = _discover_rails() if IS_JETSON else []
        self._samples: list[tuple[float, float, dict, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    # context manager
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self):
        self._t0 = time.perf_counter()
        if IS_JETSON and self._rails:
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        # if not on Jetson we synthesize at stop() based on elapsed time

    def _sample_loop(self):
        dt = self.period_ms / 1000.0
        while not self._stop.is_set():
            now = time.perf_counter() - self._t0
            total = 0.0
            rails = {}
            for name, curr_p, volt_p in self._rails:
                mA = _read_int(curr_p)
                mV = _read_int(volt_p)
                mW = (mA * mV) / 1000.0 if (mA == mA and mV == mV) else float("nan")
                rails[name] = mW
                if mW == mW:
                    total += mW
            self._samples.append((now, total, rails, _read_max_temp()))
            time.sleep(dt)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def trace(self) -> PowerTrace:
        if not self._samples:
            elapsed = max(time.perf_counter() - self._t0, 0.2)
            return _synth_trace(elapsed, self.period_ms, self.cfg)
        t = np.array([s[0] for s in self._samples])
        p = np.array([s[1] for s in self._samples])
        rail_names = self._samples[0][2].keys()
        rails = {rn: np.array([s[2].get(rn, np.nan) for s in self._samples]) for rn in rail_names}
        temp = np.array([s[3] for s in self._samples])
        return PowerTrace(t=t, p_total=p, rails=rails, temp=temp,
                          period_ms=self.period_ms, source="ina3221")


def tegrastats_available() -> bool:
    return IS_JETSON and subprocess.run(["which", "tegrastats"],
                                        capture_output=True).returncode == 0


if __name__ == "__main__":
    # quick self-test (synthetic on a laptop)
    with PowerLogger(period_ms=5) as log:
        time.sleep(0.5)
    tr = log.trace()
    print(f"source={tr.source} samples={len(tr.t)} dur={tr.duration_s():.2f}s "
          f"peak={tr.p_total.max():.0f} mW temp_max={np.nanmax(tr.temp):.1f} C")
