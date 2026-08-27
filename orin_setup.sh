#!/usr/bin/env bash
# orin_setup.sh  —  scan the Orin, then install a working PyTorch (no libcudss error).
# Run on the Orin with:   bash orin_setup.sh
# Safe: no apt, no sudo needed. Installs into your user space (~/.local) only.

set -uo pipefail

echo "==================== ORIN SCAN ===================="
echo "[python]        $(python3 --version 2>&1)"
echo "[L4T/JetPack]   $(cat /etc/nv_tegra_release 2>/dev/null | head -1)"
echo
echo "[torch/vision already installed?]"
python3 -m pip list 2>/dev/null | grep -iE "^torch|torchvision|torchaudio" || echo "   (none installed)"
echo
echo "[any torch wheels sitting on disk from before?]"
find "$HOME" -maxdepth 5 -iname "torch*.whl" 2>/dev/null | head || echo "   (none found)"
echo
echo "[does torch import RIGHT NOW?]"
python3 -c "import torch; print('   OK:', torch.__version__, 'cuda=', torch.cuda.is_available())" 2>&1 | head -3
echo
echo "[power modes on THIS board]"
grep -i "POWER_MODEL ID" /etc/nvpmodel.conf 2>/dev/null || echo "   (nvpmodel.conf not found)"
echo
echo "[INA3221 power rails visible?]"
ls /sys/class/hwmon/hwmon*/ 2>/dev/null | grep -iE "in[0-9]+_label|curr[0-9]+|power" | head || echo "   (check manually later)"
echo "==================================================="
echo

echo ">>> Installing PyTorch 2.8.0 + torchvision 0.23.0 (last versions before the"
echo ">>> libcudss dependency -- these WORK on JetPack 6.2). No URLs to type."
echo
python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null
python3 -m pip install --user --no-cache-dir \
    torch==2.8.0 torchvision==0.23.0 \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
echo

echo ">>> VERIFY  (you want to see:  2.8.0  cuda_available= True )"
python3 -c "import torch; print('RESULT:', torch.__version__, 'cuda_available=', torch.cuda.is_available())"
echo
echo ">>> Done. If RESULT says True, paste me the whole SCAN section + this line."
