#!/usr/bin/env bash
# run_all.sh — full RQ1 -> RQ3 pipeline on the Jetson Orin, one command.
# Produces: data/waveforms.csv (+ per-time traces), per-epoch training power,
#           predictor (RQ2), optimizer (RQ3), 3D Pareto, and every figure.
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
echo ">>> repo: $PWD"

# 0. this board's torch wheel is built for numpy 1.x -> keep numpy < 2
python3 -m pip install --user --no-cache-dir "numpy==1.26.4" >/dev/null 2>&1 && echo ">>> numpy pinned 1.26.4"

# 1. lock power mode + clocks (one sudo prompt). Rails are read WITHOUT sudo, so
#    the pipeline itself runs as your normal user (keeps --user torch visible).
echo ">>> caching sudo for nvpmodel switching; enter your password if asked:"
sudo -v
sudo nvpmodel -m 2 || true       # MAXN_SUPER on Orin Nano Super
sudo jetson_clocks || true
echo

# 2. RQ1 sweep — inference + training, real per-time traces saved to results/traces/
python3 run_sweep.py --iters 200 --repeats 2 --period-ms 5 \
    --modes inference train \
    --models resnet18 mobilenet_v3_large \
    --precisions FP16 \
    --batch-sizes 1 4 8 \
    --power-modes 15W 25W MAXN \
    --out data/waveforms.csv

# 3. Real epoch training -> per-epoch power (fig6) + training traces (fig5)
#    (first run downloads CIFAR-10 ~170MB once; max-steps keeps each epoch short)
python3 train_capture.py --model resnet18          --epochs 3 --batch-size 64 \
    --power-mode MAXN --max-steps-per-epoch 60 --imgsz 224
python3 train_capture.py --model mobilenet_v3_large --epochs 3 --batch-size 64 \
    --power-mode MAXN --max-steps-per-epoch 60 --imgsz 224

# 4. RQ2 predictor (real is_train now) + RQ3 optimizer
python3 -m predictor.train_predictor --data data/waveforms.csv --out results/predictor.pkl
python3 -m optimizer.optimize --data data/waveforms.csv --peak-w 50 || true

# 5. 3D Pareto + full figure deck
python3 pareto_3d.py --data data/waveforms.csv --save results/pareto_3d.png --no-show || true
python3 make_figures.py

echo
echo "==================== DONE ===================="
python3 - <<'PY'
import pandas as pd, os, glob
d = pd.read_csv("data/waveforms.csv")
print("waveforms rows :", len(d), "| trace_source:", list(d["trace_source"].unique()))
print("peak_w range   :", round(d.p_peak_w.min(),1), "-", round(d.p_peak_w.max(),1), "W  (sane Orin range ~5-40W)")
figs = [os.path.basename(f) for f in sorted(glob.glob("results/figures/*.png"))]
print("figures        :", figs)
print("3D pareto      :", "results/pareto_3d.png" if os.path.exists("results/pareto_3d.png") else "MISSING")
print("per-epoch csv  :", "data/train_epochs.csv" if os.path.exists("data/train_epochs.csv") else "MISSING")
PY
echo "=============================================="
echo ">>> Then bring results back to your PC:  git add -A && git commit -m 'orin run' && git push"
