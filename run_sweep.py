"""
run_sweep.py — the RQ1 data-collection driver.

For every DeployConfig in the sweep:
  1. (on Orin) set the power mode via nvpmodel,
  2. load the model, warm up,
  3. start the PowerLogger, run N timed inferences (or training steps), stop it,
  4. extract the waveform features,
  5. append one row {config fields + waveform features} to the dataset CSV,
  6. SAVE the raw per-time trace to results/traces/<tag>_r<rep>.csv so the
     power-over-time and energy-over-time figures use REAL data (not synthetic).

Examples:
    python run_sweep.py --smoke --modes inference train
    python run_sweep.py --iters 200 --modes inference train \
        --models resnet18 mobilenet_v3_large --precisions FP16 \
        --batch-sizes 1 4 8 --power-modes 15W 25W MAXN --out data/waveforms.csv
"""
from __future__ import annotations
import argparse
import os
import subprocess
import time

import numpy as np
import pandas as pd

import config
from models.base import make_runner
from power.telemetry import PowerLogger, IS_JETSON
from power.waveform_features import extract_features, WaveformFeatures

# Power-mode name -> nvpmodel id. DEFAULT is the Jetson Orin Nano (Super) layout
# confirmed via `nvpmodel -q` (15W=0, 25W=1, MAXN_SUPER=2). Verify on YOUR board
# with:  grep POWER_MODEL /etc/nvpmodel.conf
NVPMODEL_IDS = {"MAXN": 2, "25W": 1, "15W": 0}

TRACE_DIR = "results/traces"


def set_power_mode(mode: str):
    if not IS_JETSON:
        return
    pid = NVPMODEL_IDS.get(mode)
    if pid is None:
        print(f"[sweep] no nvpmodel id for '{mode}'; leaving current mode")
        return
    try:
        subprocess.run(["sudo", "nvpmodel", "-m", str(pid)], check=False)
        time.sleep(2)  # let clocks settle
    except Exception as e:
        print(f"[sweep] nvpmodel failed ({e}); leaving current mode")


def _save_trace(trace, cfg, rep: int):
    os.makedirs(TRACE_DIR, exist_ok=True)
    pd.DataFrame({
        "t_s": trace.t,
        "power_w": trace.p_total / 1000.0,
        "temp_c": (trace.temp if trace.temp is not None else np.full(len(trace.t), np.nan)),
    }).to_csv(f"{TRACE_DIR}/{cfg.tag()}_r{rep}.csv", index=False)


def measure_one(cfg, iters, warmup, period_ms, rep=0, save_traces=True) -> dict:
    runner = make_runner(cfg)
    runner.load()
    if warmup:
        (runner.train if cfg.mode == "train" else runner.run)(warmup)  # exclude cold-start
    with PowerLogger(period_ms=period_ms, cfg=cfg) as logger:
        res = runner.train(iters) if cfg.mode == "train" else runner.run(iters)
    trace = logger.trace()
    feats = extract_features(trace, n_inferences=res.n_inferences)
    if save_traces:
        _save_trace(trace, cfg, rep)

    row = cfg.as_row()
    row.update(feats.as_row())
    row["backend"] = res.backend
    row["throughput_infps"] = res.n_inferences / max(res.wall_s, 1e-9)
    row["trace_source"] = trace.source
    row.update({k: v for k, v in res.extra.items() if isinstance(v, (int, float))})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/waveforms.csv")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--period-ms", type=float, default=5.0)
    ap.add_argument("--repeats", type=int, default=1, help="repeat each config for variance")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--precisions", nargs="*", default=None)
    ap.add_argument("--batch-sizes", nargs="*", type=int, default=None)
    ap.add_argument("--power-modes", nargs="*", default=None)
    ap.add_argument("--modes", nargs="*", default=None,
                    help='workload modes: "inference", "train", or both')
    ap.add_argument("--save-traces", dest="save_traces", action="store_true", default=True)
    ap.add_argument("--no-save-traces", dest="save_traces", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="tiny fast sweep")
    args = ap.parse_args()

    spec = config.SweepSpec()
    if args.smoke:
        spec.models = ["resnet18", "llama3.2-1b"]
        spec.precisions = ["FP16", "INT8"]
        spec.batch_sizes = [1, 4]
        spec.power_modes = ["25W"]
        spec.llm_gen_tokens = [16]
        args.iters, args.warmup = 8, 2
    if args.models: spec.models = args.models
    if args.precisions: spec.precisions = args.precisions
    if args.batch_sizes: spec.batch_sizes = args.batch_sizes
    if args.power_modes: spec.power_modes = args.power_modes
    if args.modes: spec.modes = args.modes

    cfgs = list(spec.iter_configs())
    print(f"[sweep] board={config.detect_board()} jetson={IS_JETSON} "
          f"configs={len(cfgs)} x{args.repeats} iters={args.iters} save_traces={args.save_traces}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows, current_mode = [], None
    for i, cfg in enumerate(cfgs, 1):
        if cfg.power_mode != current_mode:
            set_power_mode(cfg.power_mode)
            current_mode = cfg.power_mode
        for rep in range(args.repeats):
            row = measure_one(cfg, args.iters, args.warmup, args.period_ms,
                              rep=rep, save_traces=args.save_traces)
            row["repeat"] = rep
            rows.append(row)
        print(f"  [{i}/{len(cfgs)}] {cfg.tag():48s} "
              f"peak={rows[-1]['p_peak_w']:.1f}W E/inf={rows[-1]['energy_per_inf_j']:.3f}J "
              f"[{rows[-1]['backend']}]")
        pd.DataFrame(rows).to_csv(args.out, index=False)  # checkpoint each config

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[sweep] wrote {len(df)} rows -> {args.out}")
    print(f"[sweep] raw traces -> {TRACE_DIR}/  (for power-over-time / energy-over-time figures)")
    print(f"[sweep] predictor targets present: {WaveformFeatures.target_names()}")


if __name__ == "__main__":
    main()
