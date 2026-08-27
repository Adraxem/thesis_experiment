"""
datacenter/scale_up.py — Part ii proof-of-concept (RQ4): turn one Orin's measured
power trace into a facility-scale power profile.

Two paths, matching the proposal:
  * OpenG2G (arXiv:2605.05519): the intended tool. It replays real GPU power traces
    through a data-center + grid model. This module EXPORTS your Orin waveform in a
    trace format you can feed to an OpenG2G data-center block, and provides a simple
    analytic scaler so you get a facility plot even before OpenG2G is wired up.
  * A block-diagram alternative in Simulink/Simscape (not implemented here).

The analytic scaler is deliberately simple and clearly labeled as a placeholder for
the real OpenG2G run: it tiles many GPUs with (a) a per-GPU count, (b) a request-
arrival jitter so their bursts don't perfectly align, and (c) an optional synchronized
fraction (training-style) that DOES align — reproducing the "everything pauses and
resumes together" spikes described in the proposal (Fig. 1 shape).

Examples:
    python -m datacenter.scale_up --trace-from-config llama3.2-1b:INT8:1:MAXN --gpus 4096
    python -m datacenter.scale_up --export-openg2g results/orin_trace.csv
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd

import config
from power.telemetry import PowerLogger, _synth_trace, PowerTrace
from power.waveform_features import extract_features


def _trace_for_config(spec: str) -> PowerTrace:
    """spec = 'model:precision:batch:powermode'."""
    model, prec, bs, pm = spec.split(":")
    cfg = config.DeployConfig(model, prec, int(bs), pm)
    # On the Orin, replace this with a real measured trace via PowerLogger + a run.
    return _synth_trace(4.0, 5.0, cfg)


def export_openg2g(trace: PowerTrace, path: str):
    """Write a two-column (time_s, power_w) trace OpenG2G can replay per-GPU."""
    df = pd.DataFrame({"time_s": trace.t, "power_w": trace.p_total / 1000.0})
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return path


def scale_facility(trace: PowerTrace, n_gpus: int, sync_fraction: float = 0.0,
                   jitter_ms: float = 40.0, seed: int = 0) -> pd.DataFrame:
    """Analytic placeholder for OpenG2G: sum n_gpus phase-shifted copies of the trace.

    sync_fraction of the GPUs fire in lockstep (training-like, additive spikes);
    the rest are randomly phase-jittered (inference-like, partially averaging out).
    """
    rng = np.random.default_rng(seed)
    t = trace.t
    p = trace.p_total / 1000.0                    # W, single GPU
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.005

    n_sync = int(n_gpus * sync_fraction)
    n_async = n_gpus - n_sync

    total = np.zeros_like(p)
    total += n_sync * p                           # perfectly aligned

    # async: shift each by a random number of samples, wrap around
    max_shift = max(int((jitter_ms / 1000.0) / dt), 1)
    # do it in groups for speed
    for _ in range(min(n_async, 512)):            # sample up to 512 distinct phases
        s = rng.integers(0, max_shift + 1)
        total += np.roll(p, s) * (n_async / min(n_async, 512))

    facility = pd.DataFrame({"time_s": t, "facility_power_w": total,
                             "facility_power_mw": total / 1e6})
    return facility


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-from-config", default="llama3.2-1b:INT8:1:MAXN")
    ap.add_argument("--gpus", type=int, default=4096)
    ap.add_argument("--sync-fraction", type=float, default=0.3,
                    help="fraction of GPUs firing in lockstep (training-like)")
    ap.add_argument("--export-openg2g", default="results/orin_trace.csv")
    ap.add_argument("--out", default="results/facility_power.csv")
    ap.add_argument("--plot", default="results/facility_power.png")
    args = ap.parse_args()

    trace = _trace_for_config(args.trace_from_config)
    feats = extract_features(trace, n_inferences=50)
    print(f"[scale] single-GPU: peak={feats.p_peak_w:.1f}W avg={feats.p_avg_w:.1f}W "
          f"peak/avg={feats.peak_to_avg:.2f}")

    export_openg2g(trace, args.export_openg2g)
    print(f"[scale] OpenG2G-ready trace -> {args.export_openg2g}")
    print(f"[scale] Wire this into an OpenG2G data-center block (arXiv:2605.05519) "
          f"for the real facility+grid run.")

    fac = scale_facility(trace, args.gpus, args.sync_fraction)
    fac.to_csv(args.out, index=False)
    peak_mw = fac["facility_power_mw"].max()
    avg_mw = fac["facility_power_mw"].mean()
    print(f"\n[scale] analytic facility ({args.gpus} GPUs, {args.sync_fraction:.0%} synced):")
    print(f"    peak={peak_mw:.2f} MW  avg={avg_mw:.2f} MW  "
          f"swing={peak_mw - fac['facility_power_mw'].min():.2f} MW")
    print(f"    -> {args.out}   (placeholder for OpenG2G; shows device choices scale up)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(fac["time_s"], fac["facility_power_mw"], lw=1.2)
        ax.set_xlabel("time (s)"); ax.set_ylabel("facility power (MW)")
        ax.set_title(f"Scaled facility power — {args.gpus} GPUs "
                     f"({args.sync_fraction:.0%} synchronized)")
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print(f"[scale] plot -> {args.plot}")
    except Exception as e:
        print(f"[scale] plot skipped: {e}")


if __name__ == "__main__":
    main()
