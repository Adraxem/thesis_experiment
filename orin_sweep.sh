#!/usr/bin/env bash
# orin_sweep.sh — collect the REAL Q1 dataset -> data/waveforms.csv
# Run on the Orin:   bash orin_sweep.sh
# Scope: the two models whose deps are installed & proven (resnet18, mobilenet_v3_large),
# FP16, inference AND training, batch 1/4/8, power modes 15W/25W/MAXN_SUPER.
# (yolo / llama / INT8 are add-ons for later — they need extra installs + a TensorRT engine.)
set -uo pipefail

# locate repo
if [ -f run_sweep.py ]; then REPO="$PWD"; else
  REPO="$(dirname "$(find "$HOME" -maxdepth 4 -name run_sweep.py 2>/dev/null | head -1)")"; fi
cd "$REPO" || { echo "repo not found"; exit 1; }
echo ">>> repo: $PWD"

# torch here was built for numpy 1.x -> pin numpy<2 so training's numpy bridge works.
# pandas/scipy/sklearn all still work with 1.26.
echo ">>> pinning numpy 1.26.4 (torch-compatible) ..."
python3 -m pip install --user --no-cache-dir "numpy==1.26.4" >/dev/null 2>&1 && echo "    done"

# correct power-mode IDs for THIS board
sed -i 's|^NVPMODEL_IDS = .*|NVPMODEL_IDS = {"MAXN": 2, "25W": 1, "15W": 0}  # Orin Nano Super|' run_sweep.py

# cache sudo so nvpmodel can switch power modes mid-sweep without stopping to ask
echo ">>> caching sudo (for nvpmodel switching) — enter your password if asked:"
sudo -v
sudo jetson_clocks || true
echo

echo ">>> SWEEP START. 2 models x FP16 x {inference,train} x batch{1,4,8} x {15W,25W,MAXN} x2 reps."
echo ">>> Saves data/waveforms.csv after EVERY config, so a crash never loses finished rows."
echo ">>> (First training config downloads CIFAR ~170MB once, if needed.)"
echo
python3 run_sweep.py \
    --iters 200 --repeats 2 --period-ms 5 \
    --modes inference train \
    --models resnet18 mobilenet_v3_large \
    --precisions FP16 \
    --batch-sizes 1 4 8 \
    --power-modes 15W 25W MAXN \
    --out data/waveforms.csv

echo
echo "==================== SWEEP SUMMARY ===================="
python3 - <<'PY'
import pandas as pd
d = pd.read_csv("data/waveforms.csv")
print("  rows collected :", len(d))
print("  trace_source   :", list(d["trace_source"].unique()), "(want just ina3221)")
print("  backends       :", list(d["backend"].unique()))
print("  peak_w range   :", round(d["p_peak_w"].min(),1), "-", round(d["p_peak_w"].max(),1), "W")
for c in ("model","mode","power_mode","batch_size","precision"):
    if c in d.columns:
        print(f"  {c:11s}:", sorted(map(str, d[c].unique())))
PY
echo "======================================================="
echo ">>> data/waveforms.csv is your real RQ1 dataset. Paste me this SUMMARY block."
