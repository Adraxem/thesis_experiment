"""
predictor/train_predictor.py — the model you TRAIN (RQ2).

Learns a mapping:  deployment config  ->  power-waveform features
so you can pick good configs without measuring every one.

Two interchangeable backends:
  * "gbm"  : sklearn GradientBoosting (one regressor per target). Default — robust
             on small datasets, no GPU needed, gives feature importances.
  * "mlp"  : a small PyTorch multi-output MLP (used if torch is installed and
             --backend mlp). This is the "small learned model" the proposal names.

Targets = WaveformFeatures.target_names():
    p_peak_w, p_avg_w, peak_to_avg, dpdt_max_w_per_s,
    dpdt_p95_w_per_s, thermal_ramp_c_per_s, energy_per_inf_j

Reports per-target MAE / R^2 with a train/test split, and saves the model +
metrics so the optimizer (RQ3) can call it.

Examples:
    python -m predictor.train_predictor                      # GBM on synthetic data
    python -m predictor.train_predictor --data data/waveforms.csv --backend mlp
"""
from __future__ import annotations
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd

import config
from power.waveform_features import WaveformFeatures

TARGETS = WaveformFeatures.target_names()


def _feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Encode each row's config columns into the numeric predictor features."""
    feats = []
    for _, r in df.iterrows():
        cfg = config.DeployConfig(
            model=r["model"], precision=r["precision"], batch_size=int(r["batch_size"]),
            power_mode=r["power_mode"], schedule=r.get("schedule", "default"), mode=r.get("mode", "inference"),
            seq_len=int(r.get("seq_len", 128)), gen_tokens=int(r.get("gen_tokens", 64)),
        )
        feats.append(cfg.to_features())
    return pd.DataFrame(feats).fillna(0.0)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        from predictor.synthetic import generate
        print(f"[predictor] {path} not found; generating synthetic dataset")
        df = generate()
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)


# --------------------------------------------------------------------------- GBM
def train_gbm(X, Y, feat_cols):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.multioutput import MultiOutputRegressor
    base = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05)
    model = MultiOutputRegressor(base)
    model.fit(X, Y)
    # aggregate feature importances across targets
    imp = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    importances = dict(sorted(zip(feat_cols, imp), key=lambda kv: -kv[1]))
    return model, {"feature_importance": importances}


# --------------------------------------------------------------------------- MLP
def train_mlp(X, Y, epochs=400, lr=1e-3):
    import torch
    import torch.nn as nn
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    xm, xs = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
    ym, ys = Yt.mean(0), Yt.std(0).clamp_min(1e-6)
    Xn, Yn = (Xt - xm) / xs, (Yt - ym) / ys
    net = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(net(Xn), Yn)
        loss.backward()
        opt.step()

    class Wrapped:
        def predict(self, Xnew):
            with torch.no_grad():
                xn = (torch.tensor(Xnew, dtype=torch.float32) - xm) / xs
                return (net(xn) * ys + ym).numpy()
    return Wrapped(), {"final_train_mse": float(loss.item())}


def evaluate(model, Xte, Yte):
    from sklearn.metrics import mean_absolute_error, r2_score
    pred = model.predict(Xte)
    out = {}
    for i, name in enumerate(TARGETS):
        out[name] = {"MAE": float(mean_absolute_error(Yte[:, i], pred[:, i])),
                     "R2": float(r2_score(Yte[:, i], pred[:, i]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_waveforms.csv")
    ap.add_argument("--backend", choices=["gbm", "mlp"], default="gbm")
    ap.add_argument("--out", default="results/predictor.pkl")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = load_data(args.data)
    Xdf = _feature_frame(df)
    feat_cols = list(Xdf.columns)
    X = Xdf.values.astype(float)
    Y = df[TARGETS].values.astype(float)

    from sklearn.model_selection import train_test_split
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=args.test_size, random_state=0)

    if args.backend == "mlp":
        try:
            model, meta = train_mlp(Xtr, Ytr)
        except Exception as e:
            print(f"[predictor] torch unavailable ({e}); using GBM")
            args.backend = "gbm"
            model, meta = train_gbm(Xtr, Ytr, feat_cols)
    else:
        model, meta = train_gbm(Xtr, Ytr, feat_cols)

    metrics = evaluate(model, Xte, Yte)
    print(f"\n[predictor] backend={args.backend}  n={len(df)}  "
          f"train={len(Xtr)} test={len(Xte)}")
    print(f"{'target':22s} {'MAE':>10s} {'R2':>8s}")
    for name, m in metrics.items():
        print(f"{name:22s} {m['MAE']:10.3f} {m['R2']:8.3f}")

    os.makedirs("results", exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump({"model": model, "feat_cols": feat_cols, "targets": TARGETS,
                     "backend": args.backend}, fh)
    with open("results/predictor_metrics.json", "w") as fh:
        json.dump({"metrics": metrics, **meta}, fh, indent=2, default=str)
    print(f"[predictor] saved -> {args.out}")
    if "feature_importance" in meta:
        top = list(meta["feature_importance"].items())[:6]
        print("[predictor] top features:", ", ".join(f"{k}={v:.2f}" for k, v in top))


if __name__ == "__main__":
    main()
