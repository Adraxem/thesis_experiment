#!/usr/bin/env bash
# run_all.sh — rigorous RQ1->RQ3 pipeline on the Jetson Orin.
# Fail-LOUD: every stage prints [OK]/[FAIL] and a final summary; nothing is hidden
# with `|| true`. The SWEEP itself is per-config resilient (an OOM config is skipped
# and reported, not fatal). Edit the SCOPE block to trade rigor vs runtime.
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"; echo ">>> repo: $PWD"

# ----------------------------- SCOPE (edit to taste) -----------------------------
MODELS="resnet18 resnet50 mobilenet_v3_large"   # supported vision models
PRECS="FP16 FP32"                               # real precision axis (INT8 needs TensorRT: later)
BATCHES="1 2 4 8"                               # add 16 for a denser surface (OOM-risky on FP32)
POWER="15W 25W MAXN"
REPS=3                                          # repeats -> error bars
ITERS=150
EPOCHS=8                                         # real training epochs (per-epoch curve)
# est: ~1-2 h. Trim MODELS/BATCHES/REPS to go faster.
# ---------------------------------------------------------------------------------

FAILS=()
step(){ echo; echo ">>> $1"; shift; if "$@"; then echo "[OK] $*"; else echo "[FAIL] $*"; FAILS+=("$*"); fi; }

# torch on this board is built for numpy 1.x
python3 -m pip install --user --no-cache-dir "numpy==1.26.4" >/dev/null 2>&1 && echo ">>> numpy pinned 1.26.4"

echo ">>> caching sudo for nvpmodel; enter password if asked:"; sudo -v
sudo nvpmodel -m 2 || true; sudo jetson_clocks || true

# fresh per-epoch file so old mock rows don't pollute fig train_per_epoch
[ -f data/train_epochs.csv ] && mv data/train_epochs.csv data/train_epochs.prev.csv 2>/dev/null || true

step "RQ1 sweep (inference + training, real traces)" \
  python3 run_sweep.py --iters "$ITERS" --repeats "$REPS" --period-ms 5 \
    --modes inference train --models $MODELS --precisions $PRECS \
    --batch-sizes $BATCHES --power-modes $POWER --out data/waveforms.csv

for M in $MODELS; do
  step "Real epoch training: $M" \
    python3 train_capture.py --model "$M" --epochs "$EPOCHS" --batch-size 64 \
      --power-mode MAXN --max-steps-per-epoch 80 --imgsz 224
done

step "RQ2 predictor (real is_train)" python3 -m predictor.train_predictor --data data/waveforms.csv --out results/predictor.pkl
step "RQ3 optimizer (50W budget)"    python3 -m optimizer.optimize --data data/waveforms.csv --peak-w 50
step "MATLAB surface export"         python3 export_for_matlab.py --data data/waveforms.csv --out results/surface_data.csv
step "Figures (honest, publication)" python3 figures.py

echo; echo "==================== DONE ===================="
python3 - <<'PY'
import pandas as pd, os, glob
d=pd.read_csv("data/waveforms.csv"); d=d[d.trace_source=='ina3221']
print("real rows      :", len(d), "| models:", sorted(d.model.unique()), "| precisions:", sorted(d.precision.unique()))
print("peak_w range   :", round(d.p_peak_w.min(),1),"-",round(d.p_peak_w.max(),1),"W")
print("figures        :", [os.path.basename(f) for f in sorted(glob.glob("results/figures/*.png"))])
PY
if [ ${#FAILS[@]} -gt 0 ]; then echo; echo ">>> STAGES THAT FAILED (not hidden):"; printf '   - %s\n' "${FAILS[@]}"; fi
echo "=============================================="
