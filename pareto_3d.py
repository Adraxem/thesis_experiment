"""
pareto_3d.py — 3D surface view of the transient-aware deployment trade-off (RQ3),
in MATLAB's look (parula colormap by default).

Axes:
    x = peak power (W)          <- the power-delivery / thermal budget lever
    y = energy per inference (J)
    z = speed score (higher = faster)   <- the objective you maximize

It fits a smooth surface z(x, y) over all candidate configs (griddata), colors it
with a MATLAB-style colormap, and overlays:
    * every config as a point,
    * the Pareto-optimal (non-dominated) configs as red markers.

Data source: the trained predictor (results/predictor.pkl) over the full config
space, or a measured CSV via --data.

Examples:
    python pareto_3d.py                       # parula surface, opens an interactive window
    python pareto_3d.py --cmap cool           # MATLAB 'cool' colormap instead
    python pareto_3d.py --data data/waveforms.csv --save results/pareto_3d.png --no-show
"""
from __future__ import annotations
import argparse

import numpy as np

from optimizer.optimize import (candidate_configs, predict_table,
                                 measured_table, pareto_front)

# MATLAB parula, as smooth anchor colors (not shipped with matplotlib).
_PARULA_ANCHORS = [
    (0.2422, 0.1504, 0.6603), (0.2810, 0.1786, 0.8258), (0.2178, 0.3282, 0.9992),
    (0.0116, 0.4433, 0.8762), (0.0779, 0.5040, 0.8384), (0.0231, 0.5732, 0.8181),
    (0.0868, 0.6446, 0.7285), (0.3501, 0.6890, 0.5266), (0.6720, 0.6874, 0.3277),
    (0.9718, 0.7350, 0.1801), (0.9769, 0.9839, 0.0805),
]


def get_cmap(name: str):
    import matplotlib
    if name.lower() == "parula":
        from matplotlib.colors import LinearSegmentedColormap
        return LinearSegmentedColormap.from_list("parula", _PARULA_ANCHORS, N=256)
    return matplotlib.colormaps[name]


def build_table(args):
    if args.data:
        df = measured_table(args.data)
    else:
        cfgs = candidate_configs(args.model, args.precisions, args.batch_sizes, args.power_modes)
        df = predict_table(args.predictor, cfgs)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", default="results/predictor.pkl")
    ap.add_argument("--data", default=None)
    ap.add_argument("--cmap", default="parula",
                    help="parula (default, MATLAB), cool, viridis, turbo, jet, ...")
    ap.add_argument("--model", nargs="*", default=None)
    ap.add_argument("--precisions", nargs="*", default=None)
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--power-modes", nargs="*", default=None)
    ap.add_argument("--grid", type=int, default=80, help="surface resolution")
    ap.add_argument("--save", default="results/pareto_3d.png")
    ap.add_argument("--no-show", action="store_true", help="save only, no window")
    ap.add_argument("--elev", type=float, default=26.0)
    ap.add_argument("--azim", type=float, default=-52.0)
    args = ap.parse_args()

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)
    from scipy.interpolate import griddata

    df = build_table(args)
    x = df["p_peak_w"].values
    y = df["energy_per_inf_j"].values
    z = df["speed_score"].values
    cmap = get_cmap(args.cmap)

    # smooth surface z(x, y)
    xi = np.linspace(x.min(), x.max(), args.grid)
    yi = np.linspace(y.min(), y.max(), args.grid)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((x, y), z, (XI, YI), method="cubic")
    ZI_lin = griddata((x, y), z, (XI, YI), method="linear")
    ZI = np.where(np.isnan(ZI), ZI_lin, ZI)     # fill cubic gaps with linear

    front = pareto_front(df, ["p_peak_w", "energy_per_inf_j", "speed_score"],
                         minimize=[True, True, False])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(XI, YI, ZI, cmap=cmap, linewidth=0, antialiased=True,
                           alpha=0.9, rcount=args.grid, ccount=args.grid)
    # all configs as faint points
    ax.scatter(x, y, z, c="k", s=8, alpha=0.25, depthshade=True)
    # Pareto front highlighted
    ax.scatter(front["p_peak_w"], front["energy_per_inf_j"], front["speed_score"],
               c="red", s=60, marker="o", edgecolors="white", linewidths=0.8,
               label=f"Pareto front (n={len(front)})", depthshade=False)

    ax.set_xlabel("Peak power (W)", labelpad=10)
    ax.set_ylabel("Energy per inference (J)", labelpad=10)
    ax.set_zlabel("Speed score", labelpad=8)
    ax.set_title(f"Transient-aware deployment trade-off  ·  {args.cmap} surface")
    ax.view_init(elev=args.elev, azim=args.azim)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.10)
    cbar.set_label("Speed score (surface)")
    ax.legend(loc="upper left")
    fig.tight_layout()

    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"[pareto3d] saved -> {args.save}  (cmap={args.cmap}, {len(df)} configs)")
    if not args.no_show:
        print("[pareto3d] opening interactive window — drag to rotate, close to exit.")
        plt.show()


if __name__ == "__main__":
    main()
