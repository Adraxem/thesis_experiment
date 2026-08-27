"""
run_sweep.py — the RQ1 data-collection driver.

For every DeployConfig in the sweep:
  1. (on Orin) set the power mode via nvpmodel,
  2. load the model, warm up,
  3. start the PowerLogger, run N timed inferences, stop the logger,
  4. extract the waveform features,
  5. append one row {config fields + waveform features} to the dataset CSV.

This CSV is the "config -> power waveform" dataset that does not exist today
(Contribution 1). It feeds the predictor (RQ2) and optimizer (RQ3).

Run on the laptop to smoke-test end-to-end (synthetic power); run on the Orin Nano
to collect real data.

Examples:
    python run_sweep.py --smoke              # tiny synthetic sweep
    python run_sweep.py --iters 200 --out data/waveforms.csv
    python run_sweep.py --models resnet18 llama3.2-1b --precisions FP16 INT8
"""
from __future__ import annotations
import argparse
import os
import subprocess
import time

import pandas as pd

import config
from models.base import make_runner
from power.telemetry import PowerLogger, IS_JETSON
from power.waveform_features import extract_features, WaveformFeatures

NVPMODEL_IDS = {"7W": 3, "15W": 2, "25W": 1, "MAXN": 0}  # confirm with `nvpmodel -q` on YOUR board!


def set_power_mode(mode: str):
    if not IS_JETSON:
        return
    pid = NVPMODEL_IDS.get(mode)
    if pid is None:
        return
    try:
        subprocess.run(["sudo", "nvpmodel", "-m", str(pid)], check=False)
        time.sleep(2)  # let clocks settle
    except Exception as e:
        print(f"[sweep] nvpmodel failed ({e}); leaving current mode")


def measure_one(cfg, iters: int, warmup: int, period_ms: float) -> dict:
    runner = make_runner(cfg)
    runner.load()
    if warmup:
        (runner.train if cfg.mode == "train" else runner.run)(warmup)  # exclude cold-start
    with PowerLogger(period_ms=period_ms, cfg=cfg) as logger:
        res = runner.train(iters) if cfg.mode == "train" else runner.run(iters)
    trace = logger.trace()
    feats = extract_features(trace, n_inferences=res.n_inferences)

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
    ap.add_argument("--smoke", action="store_true", help="tiny fast sweep")
    args = ap.parse_args()

    spec = config.SweepSpec()
    if args.smoke:
        spec.models = ["resnet18", "llama3.2-1b"]
        spec.precisions = ["FP16", "INT8"]
        spec.batch_sizes = [1, 4]
        spec.power_modes = ["25W"]
        spec.llm_gen_tokens = [16]     # keep the laptop smoke test fast
        args.iters, args.warmup = 8, 2
    if args.models: spec.models = args.models
    if args.precisions: spec.precisions = args.precisions
    if args.batch_sizes: spec.batch_sizes = args.batch_sizes
    if args.power_modes: spec.power_modes = args.power_modes
    if args.modes: spec.modes = args.modes

    cfgs = list(spec.iter_configs())
    print(f"[sweep] board={config.detect_board()} jetson={IS_JETSON} "
          f"configs={len(cfgs)} x{args.repeats} iters={args.iters}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows, current_mode = [], None
    for i, cfg in enumerate(cfgs, 1):
        if cfg.power_mode != current_mode:
            set_power_mode(cfg.power_mode)
            current_mode = cfg.power_mode
        for rep in range(args.repeats):
            row = measure_one(cfg, args.iters, args.warmup, args.period_ms)
            row["repeat"] = rep
            rows.append(row)
        print(f"  [{i}/{len(cfgs)}] {cfg.tag():48s} "
              f"peak={rows[-1]['p_peak_w']:.1f}W E/inf={rows[-1]['energy_per_inf_j']:.3f}J "
              f"[{rows[-1]['backend']}]")
        pd.DataFrame(rows).to_csv(args.out, index=False)  # checkpoint each config

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"[sweep] wrote {len(df)} rows -> {args.out}")
    print(f"[sweep] predictor targets present: {WaveformFeatures.target_names()}")


if __name__ == "__main__":
    main()
