"""
power/telemetry.py — high-rate power/thermal sampling on the Jetson Orin (RQ1).

Reads the on-board INA3221 rails via sysfs (fast, ~ms). On a non-Jetson host
(laptop / CI) it produces a *synthetic* waveform so the pipeline runs end-to-end
before hardware.

CORRECTNESS NOTE (fixed):
The INA3221 exposes several channels. On the Orin Nano these are typically
VDD_IN (the TOTAL board input) plus VDD_CPU_GPU_CV and VDD_SOC (its components).
VDD_IN already equals ~the sum of the component rails, so summing ALL discovered
rails double-counts total power (you get ~2x reality). We therefore:
  1) de-duplicate rails (the same physical channel can appear via several sysfs
     symlinks), and
  2) if a TOTAL/input rail is present, report it ALONE as p_total; otherwise sum
     only the component ("leaf") rails.
Per-rail values are always kept in the trace for inspection.

Sampling note: INA3221 samples at ~ms, so we capture PEAK power and THERMAL
transients, not us-scale di/dt. That caveat is intentional and documented.
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

# A rail whose label denotes the TOTAL board input. It already equals ~the sum of
# the component rails, so it must NOT be added to them. If found, we use it alone.
_TOTAL_RAIL_RE = re.compile(r"(VDD_IN|POM.*IN|_IN$|^VIN$|TOTAL|SUM|SYS5V|CVB.*IN)", re.I)


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
def _discover_rails() -> list:
    """Return [(label, curr_input_path, volt_input_path), ...], de-duplicated.

    Multiple globs can resolve to the SAME hwmon device via different sysfs
    symlinks; we key on the real path of the current file so each physical
    channel is listed exactly once (this alone removes one source of the
    double-counted total)."""
    found, seen = [], set()
    for pattern in _HWMON_GLOBS:
        for base in glob.glob(pattern):
            for lab in sorted(glob.glob(os.path.join(base, "*_label"))):
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
                if not (os.path.exists(curr) and os.path.exists(volt)):
                    continue
                key = os.path.realpath(curr)
                if key in seen:
                    continue
                seen.add(key)
                found.append((name, curr, volt))
    return found


def _select_total_rail(rails) -> str | None:
    """If a TOTAL/input rail exists, return its label (used alone as p_total)."""
    for name, _c, _v in rails:
        if _TOTAL_RAIL_RE.search(name or ""):
            return name
    return None


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
        steps = 10
        phase = np.linspace(0, steps * 2 * np.pi, n)
        compute = 0.82 + 0.10 * (np.sin(phase) > -0.7)
        sync_dip = 0.35 * (np.abs(np.sin(phase / 2.0)) > 0.985)
        level = np.clip(compute - sync_dip, 0.25, 1.0)
        p = floor + (peak - floor) * level
    elif is_llm:
        p[: n // 6] = peak * (0.9 + 0.1 * rng.random(n // 6))
        ripple = 0.6 + 0.25 * np.sin(np.linspace(0, 20 * np.pi, n - n // 6))
        p[n // 6:] = floor + (peak - floor) * ripple
    else:
        burst = (np.sin(np.linspace(0, 12 * np.pi, n)) > 0.2).astype(float)
        p = floor + (peak - floor) * burst

    p += rng.normal(0, peak * 0.03, n)
    p = np.clip(p, floor * 0.8, None)

    temp0, tmax = 38.0, 38.0 + (0.55 if is_train else 0.35) * budget
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
        self._total_rail = _select_total_rail(self._rails) if self._rails else None
        self._samples = []
        self._stop = threading.Event()
        self._thread = None
        self._t0 = 0.0
        if IS_JETSON and self._rails:
            names = [r[0] for r in self._rails]
            print(f"[power] rails={names}  total_rail={self._total_rail or 'SUM(leaf rails)'}")

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

    def _sample_loop(self):
        dt = self.period_ms / 1000.0
        while not self._stop.is_set():
            now = time.perf_counter() - self._t0
            rails = {}
            for name, curr_p, volt_p in self._rails:
                mA = _read_int(curr_p)
                mV = _read_int(volt_p)
                mW = (mA * mV) / 1000.0 if (mA == mA and mV == mV) else float("nan")
                rails[name] = mW
            # p_total: the TOTAL rail alone if present (avoids double-counting),
            # else the sum of the component rails.
            if self._total_rail is not None:
                total = rails.get(self._total_rail, float("nan"))
                if not (total == total):
                    total = sum(v for v in rails.values() if v == v)
            else:
                total = sum(v for v in rails.values() if v == v)
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
    with PowerLogger(period_ms=5) as log:
        time.sleep(0.5)
    tr = log.trace()
    print(f"source={tr.source} samples={len(tr.t)} dur={tr.duration_s():.2f}s "
          f"peak={tr.p_total.max():.0f} mW")
