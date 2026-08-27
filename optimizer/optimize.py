"""
optimizer/optimize.py — the transient-/peak-aware deployment picker (RQ3).

Given a budget (a peak-power ceiling and/or an energy-per-inference ceiling), it:
  1. enumerates candidate configs,
  2. predicts their waveform features with the trained predictor (RQ2)
     (or uses measured rows if you pass --data),
  3. filters to configs that satisfy the budget,
  4. ranks the survivors by an objective (throughput / speed proxy),
  5. computes the PARETO FRONT over (peak_power, energy_per_inf, speed) and
     highlights where minimizing peak power CONFLICTS with minimizing energy
     — the peak-vs-energy paradox the thesis is about.

Outputs a ranked table + a Pareto CSV + a plot.

Examples:
    python -m optimizer.optimize --peak-w 18 --energy-j 1.0
    python -m optimizer.optimize --model llama3.2-1b --peak-w 20
    python -m optimizer.optimize --data data/waveforms.csv   # use measured rows
"""
from __future__ import annotations
import argparse
import os
import pickle

import numpy as np
import pandas as pd

import config
from power.waveform_features import WaveformFeatures

TARGETS = WaveformFeatures.target_names()


def candidate_configs(models=None, precisions=None, batch_sizes=None, power_modes=None):
    spec = config.SweepSpec(
        models=models or config.MODELS,
        precisions=precisions or ["FP16", "INT8", "FP8", "INT4"],
        batch_sizes=batch_sizes or [1, 2, 4, 8, 16],
        power_modes=power_modes or config.DEFAULT_POWER_MODES,
    )
    return list(spec.iter_configs())


def predict_table(predictor_path: str, cfgs) -> pd.DataFrame:
    with open(predictor_path, "rb") as fh:
        bundle = pickle.load(fh)
    model, feat_cols = bundle["model"], bundle["feat_cols"]
    X = pd.DataFrame([c.to_features() for c in cfgs]).reindex(columns=feat_cols).fillna(0.0).values
    pred = model.predict(X)
    rows = []
    for c, p in zip(cfgs, pred):
        row = c.as_row()
        row.update({t: float(v) for t, v in zip(TARGETS, p)})
        rows.append(row)
    df = pd.DataFrame(rows)
    # a simple speed proxy: inverse energy-per-inf is NOT speed; use a latency model
    # from throughput if available, else rank by -energy (lower is better).
    df["speed_score"] = 1.0 / df["energy_per_inf_j"].clip(lower=1e-6)
    return df


def measured_table(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if "throughput_infps" in df:
        df["speed_score"] = df["throughput_infps"]
    else:
        df["speed_score"] = 1.0 / df["energy_per_inf_j"].clip(lower=1e-6)
    return df


def pareto_front(df: pd.DataFrame, cols, minimize) -> pd.DataFrame:
    """Return the non-dominated rows over `cols` (minimize[i] True=lower better)."""
    pts = df[cols].values.copy()
    for i, mn in enumerate(minimize):
        if not mn:
            pts[:, i] = -pts[:, i]
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        dominated = np.all(pts <= pts[i], axis=1) & np.any(pts < pts[i], axis=1)
        keep[dominated] = False
    return df[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="results/predictor.pkl")
    ap.add_argument("--data", default=None, help="use measured CSV instead of predictor")
    ap.add_argument("--peak-w", type=float, default=None, help="peak power ceiling (W)")
    ap.add_argument("--energy-j", type=float, default=None, help="energy/inf ceiling (J)")
    ap.add_argument("--thermal", type=float, default=None, help="max thermal ramp (C/s)")
    ap.add_argument("--model", nargs="*", default=None)
    ap.add_argument("--precisions", nargs="*", default=None)
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--power-modes", nargs="*", default=None)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--plot", default="results/pareto.png")
    args = ap.parse_args()

    if args.data:
        df = measured_table(args.data)
    else:
        cfgs = candidate_configs(args.model, args.precisions, args.batch_sizes, args.power_modes)
        df = predict_table(args.predictor, cfgs)

    total = len(df)
    if args.peak_w is not None:
        df = df[df["p_peak_w"] <= args.peak_w]
    if args.energy_j is not None:
        df = df[df["energy_per_inf_j"] <= args.energy_j]
    if args.thermal is not None:
        df = df[df["thermal_ramp_c_per_s"] <= args.thermal]

    print(f"[optimizer] {len(df)}/{total} configs satisfy the budget "
          f"(peak<={args.peak_w} W, energy<={args.energy_j} J, ramp<={args.thermal} C/s)")
    if df.empty:
        print("[optimizer] no config meets the budget — relax a constraint.")
        return

    ranked = df.sort_values("speed_score", ascending=False)
    show_cols = ["model", "precision", "batch_size", "power_mode",
                 "p_peak_w", "energy_per_inf_j", "thermal_ramp_c_per_s", "speed_score"]
    print("\n[optimizer] best configs under budget (ranked by speed):")
    print(ranked[show_cols].head(args.top).to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))

    # Pareto over the peak-vs-energy-vs-speed trade-off
    front = pareto_front(df, ["p_peak_w", "energy_per_inf_j", "speed_score"],
                         minimize=[True, True, False])
    os.makedirs("results", exist_ok=True)
    front.to_csv("results/pareto_front.csv", index=False)
    print(f"\n[optimizer] Pareto front: {len(front)} non-dominated configs "
          f"-> results/pareto_front.csv")

    # Peak-vs-energy paradox: the config with min peak vs the one with min energy
    lo_peak = df.loc[df["p_peak_w"].idxmin()]
    lo_energy = df.loc[df["energy_per_inf_j"].idxmin()]
    if lo_peak["model"] == lo_energy["model"]:
        same = (abs(lo_peak["p_peak_w"] - lo_energy["p_peak_w"]) < 1e-6)
        print("\n[optimizer] peak-vs-energy check:")
        print(f"    min-peak   : {lo_peak['precision']} b{int(lo_peak['batch_size'])} "
              f"peak={lo_peak['p_peak_w']:.2f}W energy={lo_peak['energy_per_inf_j']:.3f}J")
        print(f"    min-energy : {lo_energy['precision']} b{int(lo_energy['batch_size'])} "
              f"peak={lo_energy['p_peak_w']:.2f}W energy={lo_energy['energy_per_inf_j']:.3f}J")
        if not same:
            print("    -> they DISAGREE: minimizing peak power does not minimize energy "
                  "(the paradox). Pick per your binding constraint.")

    _plot(df, front, args.plot)


def _plot(df, front, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(df["p_peak_w"], df["energy_per_inf_j"],
                        c=df["speed_score"], cmap="viridis", s=28, alpha=0.6)
        ax.scatter(front["p_peak_w"], front["energy_per_inf_j"],
                   edgecolors="red", facecolors="none", s=90, linewidths=1.6,
                   label="Pareto front")
        ax.set_xlabel("Peak power (W)")
        ax.set_ylabel("Energy per inference (J)")
        ax.set_title("Transient-aware deployment: peak vs energy")
        fig.colorbar(sc, label="speed score (higher=faster)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=130)
        print(f"[optimizer] plot -> {path}")
    except Exception as e:
        print(f"[optimizer] plot skipped: {e}")


if __name__ == "__main__":
    main()
