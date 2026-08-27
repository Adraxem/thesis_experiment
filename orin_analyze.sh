#!/usr/bin/env bash
# orin_analyze.sh — turn data/waveforms.csv into RQ2 + RQ3 + all figures.
# Run on the Orin:   bash orin_analyze.sh
set -uo pipefail

if [ -f run_sweep.py ]; then REPO="$PWD"; else
  REPO="$(dirname "$(find "$HOME" -maxdepth 4 -name run_sweep.py 2>/dev/null | head -1)")"; fi
cd "$REPO" || { echo "repo not found"; exit 1; }
echo ">>> repo: $PWD"
DATA=data/waveforms.csv
mkdir -p results results/figures

# make_figures only trusts measured data at >=100 rows; you have 72 real ones.
# Lower the guard so YOUR data (not the synthetic set) is what gets plotted.
sed -i 's/def _dataset(min_rows=100)/def _dataset(min_rows=40)/' make_figures.py

echo; echo ">>> [1/5] RQ2 — training the power predictor on real data ..."
python3 -m predictor.train_predictor --data "$DATA" --out results/predictor.pkl

echo; echo ">>> [2/5] RQ3 — optimizing under a 50 W peak budget ..."
python3 -m optimizer.optimize --data "$DATA" --peak-w 50 || true

echo; echo ">>> [3/5] the 3D trade-off surface (peak x energy x speed) ..."
python3 pareto_3d.py --data "$DATA" --save results/pareto_3d.png --no-show || true

echo; echo ">>> [4/5] the figure deck (peak-vs-energy, precision, predictor fit, waveforms) ..."
python3 make_figures.py || true

echo; echo ">>> [5/5] exporting the surface for MATLAB ..."
python3 export_for_matlab.py --data "$DATA" --out results/surface_data.csv || true

echo
echo "==================== RESULTS ===================="
echo "[predictor R^2 per target]"
python3 - <<'PY'
import json, os
p="results/predictor_metrics.json"
if os.path.exists(p):
    m=json.load(open(p))
    for k,v in (m.get("r2") or m).items():
        try: print(f"   {k:24s} R2={float(v):+.3f}")
        except: pass
else: print("   (metrics file not found)")
PY
echo
echo "[files produced]"
ls -1 results/*.png results/figures/*.png results/*.csv results/*.pkl 2>/dev/null
echo "================================================="
echo ">>> Paste me the R^2 block + the file list. Then we scp results/ back to your desktop."
