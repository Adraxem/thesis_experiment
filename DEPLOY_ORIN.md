# Deploying to the Jetson Orin Nano

The same code you ran on your PC runs on the Orin unchanged — the difference is that
on the Orin it reads the **real INA3221 power sensor** instead of the synthetic
generator. Your job here is to (1) get the code onto the board, (2) install the
Jetson-specific ML stack, (3) fix the power-mode IDs, then run the sweep.

Important reality check: real power only means something if a **real model** is
running. If torch/TensorRT aren't installed, the model falls back to a timing stub
(a sleep) and you'd just be measuring idle power. So the ML stack below matters.

---

## 0. Identify your board and JetPack (run ON the Orin)
```bash
cat /etc/nv_tegra_release          # L4T / JetPack version
cat /proc/device-tree/model        # e.g. "NVIDIA Jetson Orin Nano ..."
python3 --version                  # JetPack 6 = 3.10, JetPack 5 = 3.8
sudo nvpmodel -q                   # <-- WRITE DOWN the mode numbers + names
```
The `nvpmodel -q` output is the single most important thing — you need those IDs
in Step 4.

---

## 1. Get the code onto the Orin
Pick whichever is easiest:

**Git (if you push the folder to GitHub):**
```bash
git clone <your-repo-url> thesis_experiment && cd thesis_experiment
```

**scp from your PC (Orin and PC on same network):**
```bash
# from your PC PowerShell, find the Orin IP with `ip addr` on the Orin first:
scp -r "C:\Users\ardac\OneDrive\Desktop\thesis_experiment" user@<orin-ip>:~/
```

**USB stick:** copy the folder over, then `cd ~/thesis_experiment`.

---

## 2. System telemetry (jtop / jetson-stats)
```bash
sudo pip3 install -U jetson-stats
sudo systemctl restart jtop.service    # or reboot
jtop                                     # confirm it shows power rails, then quit
```

## 3. Python environment (see the system TensorRT)
Use a venv WITH system packages so the JetPack-provided TensorRT/pycuda are visible:
```bash
python3 -m venv --system-site-packages ~/thesisenv
source ~/thesisenv/bin/activate
pip install -U pip
pip install numpy pandas scikit-learn matplotlib scipy
```

## 4. PyTorch + friends (the Jetson-specific part)
Do NOT `pip install torch` — the generic wheel has no CUDA for Jetson. Two options:

**Option A — Jetson wheels (native install).** For JetPack 6, install from the
Jetson AI Lab index (matches your L4T). Example:
```bash
pip install --no-cache-dir torch torchvision \
  --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126   # <-- match YOUR JetPack/CUDA
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# then:
pip install transformers peft accelerate
pip install ultralytics --no-deps        # avoid it pulling a non-Jetson torch
pip install onnx
```
(Confirm the exact index URL for your JetPack at the NVIDIA Jetson forums / Jetson
AI Lab — the `jp6/cu126` part changes per JetPack.)

**Option B — NVIDIA container (easiest, most reliable).** The `l4t-ml` / `l4t-pytorch`
image already has torch, torchvision, and TensorRT built for your board:
```bash
sudo docker run -it --rm --runtime nvidia --network host \
  -v ~/thesis_experiment:/work -w /work \
  nvcr.io/nvidia/l4t-pytorch:r36.2.0-pth2.2-py3    # <-- pick the tag for your L4T
# inside the container:
pip install pandas scikit-learn matplotlib scipy transformers peft accelerate
```

TensorRT and pycuda ship with JetPack — never pip-install them.

## 5. Fix the power-mode IDs  (REQUIRED — do not skip)
Open `run_sweep.py` and edit the `NVPMODEL_IDS` dict near the top to match what
`sudo nvpmodel -q` printed on YOUR board. The defaults are a guess and are probably
wrong for your Nano. Example if `-q` showed 0=25W(MAXN), 1=15W, 2=7W:
```python
NVPMODEL_IDS = {"MAXN": 0, "15W": 1, "7W": 2}
```
Then set the sweep's power modes to the ones your board actually has, e.g.
`--power-modes 7W 15W MAXN`.

## 6. Sanity check (one config, prove sensors are real)
```bash
# lock clocks for repeatable numbers (optional but recommended):
sudo jetson_clocks

# run a tiny real sweep. nvpmodel switching needs sudo — either run with sudo, or
# set one mode by hand and skip switching:
sudo nvpmodel -m 0
python3 run_sweep.py --iters 50 --warmup 10 \
    --models resnet18 --precisions FP16 --batch-sizes 1 \
    --power-modes MAXN --out data/probe.csv

# CONFIRM IT'S REAL:
python3 - <<'PY'
import pandas as pd
d = pd.read_csv("data/probe.csv")
print("trace_source:", d["trace_source"].unique())   # must say 'ina3221', NOT 'mock'
print("backend:", d["backend"].unique())              # must be torch-cuda..., NOT 'stub'
print("peak_w:", d["p_peak_w"].round(2).tolist())
PY
```
If `trace_source` is `mock` → the code thinks it's not a Jetson (check
`platform.machine()` is `aarch64`). If `backend` is `stub` → torch didn't install
with CUDA; fix Step 4.

## 7. The real sweep (this is your dataset)
```bash
# inference + training, several repeats for variance. sudo so nvpmodel can switch.
sudo -E python3 run_sweep.py \
    --iters 300 --repeats 3 --period-ms 5 \
    --modes inference train \
    --models resnet18 mobilenet_v3_large yolov8n llama3.2-1b \
    --precisions FP16 INT8 \
    --batch-sizes 1 4 8 \
    --power-modes 15W MAXN \
    --out data/waveforms.csv
```
This writes/append-checkpoints `data/waveforms.csv` after every config, so a crash
mid-sweep doesn't lose finished rows. Start SMALL (few models/precisions) and grow.

Note on the LLM: for real INT4/INT8 *decode* power, later swap the HF model in
`models/llm.py` for a TensorRT-LLM engine or a llama.cpp GGUF — the HF fp16 path
works but isn't the true low-precision number.

## 8. Analyze on the real data
```bash
python3 -m predictor.train_predictor --data data/waveforms.csv   # RQ2 on REAL data
python3 -m optimizer.optimize --data data/waveforms.csv --peak-w 15
python3 make_figures.py                        # auto-uses waveforms.csv once >=100 rows
python3 export_for_matlab.py --data data/waveforms.csv
```
The predictor's R² on THIS data is your actual Q2 result (unlike the synthetic ~0.99).

## 9. Bring results back to your PC
```bash
# from your PC:
scp -r user@<orin-ip>:~/thesis_experiment/data   "C:\Users\ardac\OneDrive\Desktop\thesis_experiment\"
scp -r user@<orin-ip>:~/thesis_experiment/results "C:\Users\ardac\OneDrive\Desktop\thesis_experiment\"
```
Then open `pareto_surface.m` in MATLAB for the interactive surface of the REAL data.

---

## Troubleshooting
- `torch.cuda.is_available()` is False → you installed a generic wheel; redo Step 4A
  with the Jetson index, or use the container (4B).
- `nvpmodel: command not found` → not on a Jetson, or PATH issue; it's in `/usr/sbin`.
- Permission denied reading rails → run the sweep with `sudo` (or `sudo -E` to keep
  your venv/env).
- tegrastats present but rails empty → some JetPacks name the hwmon paths differently;
  `ls /sys/class/hwmon/hwmon*/` and check `*_label` files, then widen the globs in
  `power/telemetry.py` (`_HWMON_GLOBS`).
- Thermal throttling mid-sweep is EXPECTED and is part of what you're studying — add
  `--repeats` and a cooldown between heavy configs if you want steady-state numbers.
