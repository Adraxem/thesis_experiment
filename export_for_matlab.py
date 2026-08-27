"""
export_for_matlab.py — dump the optimizer's config table to a CSV that the MATLAB
script (pareto_surface.m) reads to draw an interactive 3D surface.

Columns: model, precision, batch_size, power_mode, p_peak_w, energy_per_inf_j,
         thermal_ramp_c_per_s, speed_score, pareto (1 = non-dominated).

Uses the trained predictor over the full config space by default, or a measured
CSV via --data.
"""
from __future__ import annotations
import argparse
import os

from optimizer.optimize import (candidate_configs, predict_table,
                                 measured_table, pareto_front)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="results/predictor.pkl")
    ap.add_argument("--data", default=None, help="use measured CSV instead of predictor")
    ap.add_argument("--out", default="results/surface_data.csv")
    args = ap.parse_args()

    if args.data:
        df = measured_table(args.data)
    else:
        df = predict_table(args.predictor, candidate_configs())

    front = pareto_front(df, ["p_peak_w", "energy_per_inf_j", "speed_score"],
                         minimize=[True, True, False])
    df = df.copy()
    df["pareto"] = df.index.isin(front.index).astype(int)

    keep = ["model", "precision", "batch_size", "power_mode", "p_peak_w",
            "energy_per_inf_j", "thermal_ramp_c_per_s", "speed_score", "pareto"]
    keep = [c for c in keep if c in df.columns]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df[keep].to_csv(args.out, index=False)
    print(f"[export] wrote {len(df)} rows ({int(df['pareto'].sum())} Pareto) -> {args.out}")
    print(f"[export] open pareto_surface.m in MATLAB to view the interactive surface.")


if __name__ == "__main__":
    main()
