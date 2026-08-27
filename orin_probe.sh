#!/usr/bin/env bash
# orin_probe.sh — prove the INA3221 sensor + CUDA backend are REAL (checkpoint 2).
# Run on the Orin:   bash orin_probe.sh
# It will ask for your password ONCE (for nvpmodel). Rails are read without sudo,
# so your --user torch stays visible.
set -uo pipefail

# 1. locate the repo (clone it if it isn't on the board yet)
HIT="$(find "$HOME" -maxdepth 4 -name run_sweep.py 2>/dev/null | head -1)"
if [ -n "$HIT" ]; then REPO="$(dirname "$HIT")"; else REPO=""; fi
if [ -z "$REPO" ] || [ ! -f "$REPO/run_sweep.py" ]; then
  echo ">>> repo not on board — cloning to ~/thesis_experiment"
  git clone https://github.com/Adraxem/thesis_experiment.git "$HOME/thesis_experiment"
  REPO="$HOME/thesis_experiment"
fi
cd "$REPO" || exit 1
echo ">>> using repo: $REPO"
echo

# 2. print the REAL power modes for this board (I need these to finish the sweep labels)
echo "==================== POWER MODES ===================="
grep -i "POWER_MODEL ID" /etc/nvpmodel.conf 2>/dev/null || echo "(nvpmodel.conf not found)"
echo "===================================================="
echo

# 3. patch the power-mode map so MAXN -> 2 (MAXN_SUPER, confirmed on your board),
#    so the probe actually runs at full power instead of the wrong default id.
sed -i 's|^NVPMODEL_IDS = .*|NVPMODEL_IDS = {"MAXN": 2, "25W": 1, "15W": 0}  # patched for Orin Nano Super|' run_sweep.py

# 4. lock to MAXN_SUPER + max clocks for a clean signal (one password prompt)
sudo nvpmodel -m 2 || true
sudo jetson_clocks || true
echo

# 5. tiny REAL probe — NO sudo here, so your --user torch is visible
python3 run_sweep.py --iters 50 --warmup 10 \
    --models resnet18 --precisions FP16 --batch-sizes 1 \
    --power-modes MAXN --out data/probe.csv

# 6. verdict
echo
echo "================ CHECKPOINT 2 VERDICT ================"
python3 - <<'PY'
import pandas as pd
d = pd.read_csv("data/probe.csv")
print("trace_source:", list(d["trace_source"].unique()), " <- want ['ina3221'], NOT ['mock']")
print("backend     :", list(d["backend"].unique()),      " <- want torch-cuda..., NOT ['stub']")
print("peak_w      :", d["p_peak_w"].round(2).tolist())
print("energy/inf J:", d["energy_per_inf_j"].round(4).tolist())
PY
echo "====================================================="
echo ">>> Paste me the POWER MODES block + this VERDICT block."
