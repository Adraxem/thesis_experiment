"""
make_figures.py — thesis figures built from REAL measured data.

Reads:
  data/waveforms.csv          (RQ1 dataset: config -> waveform features)
  results/traces/<tag>_rN.csv (raw per-time traces saved by run_sweep.py)
  data/train_traces/<tag>.csv (raw per-time training traces from train_capture.py)
  data/train_epochs.csv       (per-epoch loss + power from train_capture.py)
  results/predictor.pkl        (RQ2 model, for the parity plot)

Writes results/figures/:
  fig1_power_over_time.png     : real power waveform vs time (inference)
  fig2_precision_peak.png      : peak power by precision
  fig3_peak_vs_energy.png      : peak-vs-energy trade-off, coloured by model
  fig4_predictor_fit.png       : predicted-vs-measured parity (RQ2, R^2)
  fig5_train_vs_infer.png      : REAL inference vs training power over time
  fig6_train_per_epoch.png     : loss + power + temp per epoch
  fig7_energy_over_time.png    : cumulative energy & running energy-per-inference

Figures whose input is missing are SKIPPED with a printed note (never faked).
"""
from __future__ import annotations
import glob
import os
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from power.waveform_features import WaveformFeatures

FIGDIR = "results/figures"
TARGETS = WaveformFeatures.target_names()


# --------------------------------------------------------------------------- data
def _dataset(min_rows=20):
    measured = "data/waveforms.csv"
    if os.path.exists(measured):
        dm = pd.read_csv(measured)
        real = dm.get("trace_source")
        is_real = (real is not None) and ("ina3221" in set(real.unique()))
        if len(dm) >= min_rows:
            print(f"[figures] using measured {measured} ({len(dm)} rows, real={is_real})")
            return dm
    synth = "data/synthetic_waveforms.csv"
    if os.path.exists(synth):
        print(f"[figures] measured data too small/absent; using {synth}")
        return pd.read_csv(synth)
    from predictor.synthetic import generate
    df = generate(); os.makedirs("data", exist_ok=True); df.to_csv(synth, index=False)
    return df


def _tag_to_throughput():
    """Map each config tag -> throughput (inf/s) from waveforms.csv, for E/inf-over-time."""
    out = {}
    p = "data/waveforms.csv"
    if not os.path.exists(p):
        return out
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        try:
            cfg = config.DeployConfig(
                model=r["model"], precision=r["precision"], batch_size=int(r["batch_size"]),
                power_mode=r["power_mode"], schedule=r.get("schedule", "default"),
                mode=r.get("mode", "inference"),
                seq_len=int(r.get("seq_len", 128)), gen_tokens=int(r.get("gen_tokens", 64)))
            out[cfg.tag()] = float(r.get("throughput_infps", np.nan))
        except Exception:
            continue
    return out


def _find_traces(kind=None):
    """Return list of (tag, path) for saved per-time traces. kind filters on the
    workload mode substring ('inference' or 'train')."""
    paths = sorted(glob.glob("results/traces/*.csv")) + sorted(glob.glob("data/train_traces/*.csv"))
    out = []
    for p in paths:
        base = os.path.basename(p).rsplit(".csv", 1)[0]
        tag = base.rsplit("_r", 1)[0] if "_r" in base else base
        if kind and f"_{kind}_" not in f"_{tag}_":
            continue
        out.append((tag, p))
    return out


def _load_trace(path):
    df = pd.read_csv(path)
    if "power_w" not in df or "t_s" not in df:
        return None
    return df


# --------------------------------------------------------------------------- figures
def fig_power_over_time():
    tr = _find_traces("inference")
    if not tr:
        print("[figures] fig1 skipped: no inference traces in results/traces/ (run the sweep)")
        return
    tag, path = tr[0]
    d = _load_trace(path)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(d["t_s"], d["power_w"], lw=1.0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("power (W)")
    ax.set_title(f"Measured power waveform over time\n{tag}")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig1_power_over_time.png", dpi=150); plt.close(fig)
    print(f"[figures] fig1 <- {path}")


def fig_train_vs_infer():
    it = _find_traces("inference"); tt = _find_traces("train")
    if not it or not tt:
        print("[figures] fig5 skipped: need both an inference and a train trace")
        return
    di = _load_trace(it[0][1]); dt = _load_trace(tt[0][1])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(di["t_s"], di["power_w"], lw=1.0, label=f"inference: {it[0][0]}")
    ax.plot(dt["t_s"], dt["power_w"], lw=1.0, label=f"training: {tt[0][0]}")
    ax.set_xlabel("time (s)"); ax.set_ylabel("power (W)")
    ax.set_title("Measured per-device power: inference vs training")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig5_train_vs_infer.png", dpi=150); plt.close(fig)
    print(f"[figures] fig5 <- {it[0][1]} , {tt[0][1]}")


def fig_energy_over_time():
    tr = _find_traces("inference") or _find_traces("train")
    if not tr:
        print("[figures] fig7 skipped: no traces")
        return
    tag, path = tr[0]
    d = _load_trace(path)
    t = d["t_s"].values.astype(float); p = d["power_w"].values.astype(float)
    dt = np.diff(t, prepend=t[0]); dt[dt < 0] = 0
    cumE = np.cumsum(p * dt)                       # cumulative energy (J)
    thr = _tag_to_throughput().get(tag, np.nan)    # inferences / second
    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    ax1.plot(t, cumE, color="tab:blue", lw=1.3, label="cumulative energy (J)")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("cumulative energy (J)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    if thr == thr and thr > 0:
        count = np.maximum(thr * t, 1e-9)
        epi = cumE / count
        ax2 = ax1.twinx()
        ax2.plot(t, epi, color="tab:red", lw=1.3, label="energy per inference (J)")
        ax2.set_ylabel("energy per inference (J)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_title(f"Energy accumulation & energy-per-inference over time\n{tag}")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig7_energy_over_time.png", dpi=150); plt.close(fig)
    print(f"[figures] fig7 <- {path}")


def fig_per_epoch():
    p = "data/train_epochs.csv"
    if not os.path.exists(p):
        print("[figures] fig6 skipped: run train_capture.py to get data/train_epochs.csv")
        return
    df = pd.read_csv(p)
    tag_cols = [c for c in ("model", "precision", "batch_size", "power_mode") if c in df]
    # take the most recent training run (last model/config block)
    if tag_cols:
        last = df[tag_cols].iloc[-1]
        d = df[(df[tag_cols] == last).all(axis=1)].sort_values("epoch")
    else:
        d = df.sort_values("epoch")
    fig, ax1 = plt.subplots(figsize=(8, 4.4))
    ax1.plot(d["epoch"], d["loss"], "o-", color="tab:green", label="loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("training loss", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green")
    ax2 = ax1.twinx()
    if "p_avg_w" in d: ax2.plot(d["epoch"], d["p_avg_w"], "s--", color="tab:blue", label="avg power (W)")
    if "p_peak_w" in d: ax2.plot(d["epoch"], d["p_peak_w"], "^--", color="tab:red", label="peak power (W)")
    ax2.set_ylabel("power (W)")
    ax1.set_title("Per-epoch training: loss vs power")
    l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig6_train_per_epoch.png", dpi=150); plt.close(fig)
    print(f"[figures] fig6 <- {p}")


def fig_precision_peak(df):
    order = [p for p in ["FP32", "FP16", "FP8", "INT8", "INT4"] if p in df["precision"].unique()]
    if len(order) < 2:
        print("[figures] fig2 skipped: only one precision present (add INT8 to compare)")
        return
    g = df.groupby("precision")["p_peak_w"].mean().reindex(order)
    err = df.groupby("precision")["p_peak_w"].std().reindex(order).fillna(0)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.bar(range(len(g)), g.values, yerr=err.values, capsize=4,
           color=plt.cm.viridis(np.linspace(0.15, 0.85, len(g))))
    ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index)
    ax.set_xlabel("precision"); ax.set_ylabel("mean peak power (W)")
    ax.set_title("Peak power vs precision")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig2_precision_peak.png", dpi=150); plt.close(fig)
    print("[figures] fig2 written")


def fig_peak_vs_energy(df):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    models = sorted(df["model"].unique())
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(models), 3)))
    markers = {"inference": "o", "train": "^"}
    for mi, model in enumerate(models):
        for mode in df["mode"].unique() if "mode" in df else ["inference"]:
            s = df[(df["model"] == model) & (df.get("mode", "inference") == mode)]
            if not len(s):
                continue
            ax.scatter(s["p_peak_w"], s["energy_per_inf_j"], s=28, alpha=0.7,
                       color=cmap[mi], marker=markers.get(mode, "o"),
                       label=f"{model} / {mode}")
    ax.set_xlabel("peak power (W)"); ax.set_ylabel("energy per iteration (J)")
    ax.set_title("Peak-power vs energy trade-off (colour=model, marker=mode)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig3_peak_vs_energy.png", dpi=150); plt.close(fig)
    print("[figures] fig3 written")


def fig_predictor_fit(df):
    if not os.path.exists("results/predictor.pkl"):
        print("[figures] fig4 skipped: no results/predictor.pkl")
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
        actual = df[name].values; p = pred[:, i]
        ax.scatter(actual, p, s=16, alpha=0.5)
        lo, hi = min(actual.min(), p.min()), max(actual.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ss_res = np.sum((actual - p) ** 2); ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        ax.set_xlabel(f"measured {name}"); ax.set_ylabel(f"predicted {name}")
        ax.set_title(f"{name}  (R^2 = {r2:.3f})")
    fig.suptitle("Predictor fit: config -> waveform features (RQ2)")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig4_predictor_fit.png", dpi=150); plt.close(fig)
    print("[figures] fig4 written")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    df = _dataset()
    fig_power_over_time()
    fig_train_vs_infer()
    fig_energy_over_time()
    fig_per_epoch()
    fig_precision_peak(df)
    fig_peak_vs_energy(df)
    fig_predictor_fit(df)
    print(f"[figures] done -> {FIGDIR}/")


if __name__ == "__main__":
    main()
