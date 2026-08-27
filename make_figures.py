"""
make_figures.py — generate the thesis figures from the dataset + trained predictor.

Writes PNGs into results/figures/:
  fig1_waveforms.png     : example power waveforms over time — vision burst vs
                           LLM prefill spike + decode ripple (the "shape" RQ1 studies)
  fig2_precision_peak.png: peak power by precision (the known quantization effect)
  fig3_peak_vs_energy.png: the peak-vs-energy trade-off / paradox scatter
  fig4_predictor_fit.png : predicted vs actual parity plots with R^2 (RQ2 quality)

The 3D surface (pareto_3d.py) and facility figure (datacenter.scale_up) are made by
their own scripts; run_all.bat calls everything.

Uses measured data/waveforms.csv if present, else data/synthetic_waveforms.csv.
"""
from __future__ import annotations
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from power.telemetry import _synth_trace
from power.waveform_features import WaveformFeatures

FIGDIR = "results/figures"
TARGETS = WaveformFeatures.target_names()


def _dataset(min_rows=100):
    """Prefer measured data once it is substantial; else the synthetic set.
    A tiny --smoke sweep (a handful of rows) should not drive the figures."""
    measured = "data/waveforms.csv"
    if os.path.exists(measured):
        dm = pd.read_csv(measured)
        if len(dm) >= min_rows:
            return dm, measured
    synth = "data/synthetic_waveforms.csv"
    if os.path.exists(synth):
        return pd.read_csv(synth), synth
    from predictor.synthetic import generate
    df = generate(); df.to_csv(synth, index=False)
    return df, synth


def fig_waveforms():
    """Two representative traces so the 'power shape' is visible."""
    vis = config.DeployConfig("resnet50", "FP16", 4, "MAXN")
    llm = config.DeployConfig("llama3.2-1b", "FP16", 1, "MAXN")
    tv = _synth_trace(3.0, 5.0, vis)
    tl = _synth_trace(3.0, 5.0, llm)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(tv.t, tv.p_total / 1000.0, lw=1.3, label="ResNet-50 (vision) — periodic bursts")
    ax.plot(tl.t, tl.p_total / 1000.0, lw=1.3, label="Llama-3.2-1B — prefill spike + decode ripple")
    ax.set_xlabel("time (s)"); ax.set_ylabel("power (W)")
    ax.set_title("Power waveform shape by workload (Jetson Orin Nano)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig1_waveforms.png", dpi=150); plt.close(fig)


def fig_train_vs_infer():
    """Same model, two regimes: inference bursts vs sustained training + sync dips."""
    infer = config.DeployConfig("resnet50", "FP16", 4, "MAXN", mode="inference")
    train = config.DeployConfig("resnet50", "FP16", 4, "MAXN", mode="train")
    ti = _synth_trace(3.0, 5.0, infer)
    tt = _synth_trace(3.0, 5.0, train)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(ti.t, ti.p_total / 1000.0, lw=1.3, label="inference (bursty, incoherent)")
    ax.plot(tt.t, tt.p_total / 1000.0, lw=1.3,
            label="training (sustained + periodic all-reduce dips)")
    ax.set_xlabel("time (s)"); ax.set_ylabel("power (W)")
    ax.set_title("Per-device power: inference vs training (ResNet-50, Orin Nano)")
    ax.legend(); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig5_train_vs_infer.png", dpi=150); plt.close(fig)


def fig_precision_peak(df):
    order = [p for p in ["FP32", "FP16", "FP8", "INT8", "INT4"] if p in df["precision"].unique()]
    g = df.groupby("precision")["p_peak_w"].mean().reindex(order)
    err = df.groupby("precision")["p_peak_w"].std().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(range(len(g)), g.values, yerr=err.values, capsize=4,
           color=plt.cm.parula(np.linspace(0.15, 0.85, len(g))) if hasattr(plt.cm, "parula")
           else plt.cm.viridis(np.linspace(0.15, 0.85, len(g))))
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index)
    ax.set_xlabel("precision"); ax.set_ylabel("mean peak power (W)")
    ax.set_title("Peak power drops with lower precision (quantization effect)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig2_precision_peak.png", dpi=150); plt.close(fig)


def fig_peak_vs_energy(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    for prec in df["precision"].unique():
        s = df[df["precision"] == prec]
        ax.scatter(s["p_peak_w"], s["energy_per_inf_j"], s=22, alpha=0.6, label=prec)
    ax.set_xlabel("peak power (W)"); ax.set_ylabel("energy per inference (J)")
    ax.set_title("Peak-power vs energy-per-inference trade-off")
    ax.legend(title="precision"); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig3_peak_vs_energy.png", dpi=150); plt.close(fig)


def fig_predictor_fit(df):
    if not os.path.exists("results/predictor.pkl"):
        print("[figures] predictor.pkl not found; skipping parity plot")
        return
    from predictor.train_predictor import _feature_frame
    with open("results/predictor.pkl", "rb") as fh:
        b = pickle.load(fh)
    X = _feature_frame(df).reindex(columns=b["feat_cols"]).fillna(0.0).values
    pred = b["model"].predict(X)
    show = ["p_peak_w", "energy_per_inf_j"]
    fig, axes = plt.subplots(1, len(show), figsize=(11, 4.6))
    for ax, name in zip(axes, show):
        i = TARGETS.index(name)
        actual = df[name].values
        p = pred[:, i]
        ax.scatter(actual, p, s=16, alpha=0.5)
        lo, hi = min(actual.min(), p.min()), max(actual.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ss_res = np.sum((actual - p) ** 2); ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        ax.set_xlabel(f"measured {name}"); ax.set_ylabel(f"predicted {name}")
        ax.set_title(f"{name}  (R^2 = {r2:.3f})")
    fig.suptitle("Predictor fit: config -> waveform features (RQ2)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig4_predictor_fit.png", dpi=150); plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df, src = _dataset()
    print(f"[figures] dataset: {src}  ({len(df)} rows)")
    fig_waveforms()
    fig_train_vs_infer()
    fig_precision_peak(df)
    fig_peak_vs_energy(df)
    fig_predictor_fit(df)
    print(f"[figures] wrote fig1..fig5 -> {FIGDIR}/")


if __name__ == "__main__":
    main()
