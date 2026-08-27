"""
power/waveform_features.py — turn a PowerTrace into the scalar features that are
the *object of study* in this thesis (RQ1): peak, transient sharpness, thermal
ramp, and energy-per-inference — not just average power.

These are exactly the targets the predictor (RQ2) learns and the optimizer (RQ3)
constrains.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict

import numpy as np

from .telemetry import PowerTrace


@dataclass
class WaveformFeatures:
    # power (Watts)
    p_peak_w: float
    p_avg_w: float
    p_idle_w: float
    p_p95_w: float
    peak_to_avg: float            # crest factor
    # transients (Watts / second) — how sharp the jumps are (proxy for di/dt)
    dpdt_max_w_per_s: float
    dpdt_p95_w_per_s: float
    transient_count: float        # # of jumps above a threshold, per second
    # thermal
    temp_max_c: float
    thermal_ramp_c_per_s: float   # slope of the heating curve
    # energy
    energy_total_j: float
    energy_per_inf_j: float
    # bookkeeping
    duration_s: float
    n_inferences: float

    def as_row(self) -> dict:
        return asdict(self)

    @staticmethod
    def target_names() -> list[str]:
        """The waveform features the predictor learns (RQ2)."""
        return ["p_peak_w", "p_avg_w", "peak_to_avg", "dpdt_max_w_per_s",
                "dpdt_p95_w_per_s", "thermal_ramp_c_per_s", "energy_per_inf_j"]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    a = np.polyfit(x, y, 1)
    return float(a[0])


def extract_features(trace: PowerTrace, n_inferences: int) -> WaveformFeatures:
    t = np.asarray(trace.t, float)
    p_w = np.asarray(trace.p_total, float) / 1000.0   # mW -> W
    n = max(int(n_inferences), 1)
    dur = max(trace.duration_s(), 1e-6)

    p_peak = float(np.nanmax(p_w))
    p_avg = float(np.nanmean(p_w))
    p_idle = float(np.nanpercentile(p_w, 5))
    p95 = float(np.nanpercentile(p_w, 95))

    # dP/dt via finite differences
    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    dpdt = np.abs(np.diff(p_w) / dt)
    dpdt = dpdt[np.isfinite(dpdt)]
    dpdt_max = float(np.max(dpdt)) if dpdt.size else 0.0
    dpdt_p95 = float(np.percentile(dpdt, 95)) if dpdt.size else 0.0
    # count sharp transients (> 25% of span within one sample), per second
    span = max(p_peak - p_idle, 1e-6)
    thr = 0.25 * span / (trace.period_ms / 1000.0)
    transient_count = float(np.sum(dpdt > thr)) / dur

    # energy = integral P dt
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    energy = float(_trapz(p_w, t)) if (_trapz and len(t) > 1) else p_avg * dur
    energy_per_inf = energy / n

    # thermal ramp: slope over the first ~60% (before plateau)
    if trace.temp is not None and len(trace.temp) > 2:
        temp = np.asarray(trace.temp, float)
        temp_max = float(np.nanmax(temp))
        k = max(3, int(len(temp) * 0.6))
        ramp = _slope(t[:k], temp[:k])
    else:
        temp_max, ramp = float("nan"), 0.0

    return WaveformFeatures(
        p_peak_w=p_peak, p_avg_w=p_avg, p_idle_w=p_idle, p_p95_w=p95,
        peak_to_avg=p_peak / max(p_avg, 1e-6),
        dpdt_max_w_per_s=dpdt_max, dpdt_p95_w_per_s=dpdt_p95,
        transient_count=transient_count,
        temp_max_c=temp_max, thermal_ramp_c_per_s=ramp,
        energy_total_j=energy, energy_per_inf_j=energy_per_inf,
        duration_s=dur, n_inferences=float(n),
    )


if __name__ == "__main__":
    from .telemetry import _synth_trace
    import config
    cfg = config.DeployConfig("llama3.2-1b", "INT8", 1, "25W")
    tr = _synth_trace(3.0, 5.0, cfg)
    f = extract_features(tr, n_inferences=20)
    for k, v in f.as_row().items():
        print(f"{k:24s} {v:10.3f}")
