# thesis_experiment

![CI](https://github.com/Adraxem/thesis_experiment/actions/workflows/ci.yml/badge.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)


Experiment scaffold for **"Managing Power Behavior of ML Inference: From Edge Device
to the Data Centers"** (Yildiz, M.Sc. proposal v4). Target device: **NVIDIA Jetson
Orin Nano**. Small LLM: **Llama-3.2-1B**.

Everything runs **today on your laptop** with a synthetic power model, and switches
to **real measurement** automatically when run on the Orin (aarch64 + INA3221 rails).
That lets you build/debug the whole pipeline before hardware time.

## Maps to your research questions

| RQ | Question | Files |
|----|----------|-------|
| Q1 | How do deployment settings shape the measured power *waveform*? | `run_sweep.py`, `power/telemetry.py`, `power/waveform_features.py` |
| Q2 | Can a small model predict the waveform from settings? | `predictor/train_predictor.py`, `predictor/synthetic.py` |
| Q3 | Given a power/energy budget, which config is best? | `optimizer/optimize.py` |
| Q4 | Feed one Orin into a data-center model for facility scale? | `datacenter/scale_up.py` (OpenG2G export) |

## The two "models to train"
1. **The inference models you benchmark** — ResNet, MobileNet, YOLOv8n, Llama-3.2-1B
   (`models/`). These are run under load so their power waveform is measured; on the
   Orin they go PyTorch → ONNX → TensorRT (FP16/INT8/…).
2. **The predictor you actually train** (`predictor/train_predictor.py`) — a small
   GBM (default) or PyTorch MLP that learns *config → waveform features*. This is the
   ML core of Q2.

## Install
```bash
pip install -r requirements.txt
# On the Orin: use the NVIDIA JetPack wheels for torch/torchvision + `sudo pip install jetson-stats`.
# TensorRT/pycuda come with JetPack — do NOT pip install them on the device.
```

## Quickstart (laptop — synthetic power)
```bash
python config.py                                   # inspect the config space
python run_sweep.py --smoke                         # tiny end-to-end sweep -> data/waveforms.csv
python -m predictor.synthetic                        # 1200-row synthetic dataset
python -m predictor.train_predictor                  # train predictor -> results/predictor.pkl
python -m optimizer.optimize --peak-w 18 --energy-j 1.0   # RQ3 picker + Pareto plot
python -m datacenter.scale_up --gpus 4096            # RQ4 facility scale-up
```

## On the Jetson Orin Nano (real data)
```bash
# 1) Confirm the power-mode IDs on YOUR board and fix them in run_sweep.py (NVPMODEL_IDS):
sudo nvpmodel -q                # lists the modes/IDs for your JetPack
sudo jetson_clocks              # (optional) lock clocks for repeatable runs

# 2) Collect the real config->waveform dataset:
python run_sweep.py --iters 300 --repeats 3 --out data/waveforms.csv

# 3) Train the predictor on measured data and run the optimizer against it:
python -m predictor.train_predictor --data data/waveforms.csv --backend mlp
python -m optimizer.optimize --data data/waveforms.csv --peak-w 20
```

## Things to plug in (marked `TODO` in code)
- **TensorRT engine build** in `models/vision.py` (ONNX export is already done) and a
  **TensorRT-LLM / llama.cpp GGUF** backend in `models/llm.py` for true INT4/INT8
  decode power.
- **Power-mode IDs** in `run_sweep.py` — verify against `nvpmodel -q`.
- **OpenG2G wiring** in `datacenter/scale_up.py` — the analytic scaler is a labeled
  placeholder; the CSV it exports is the real OpenG2G input.

## Outputs
- `data/waveforms.csv` — the config→waveform dataset (**Contribution 1**).
- `results/predictor.pkl` + `results/predictor_metrics.json` — trained predictor + accuracy.
- `results/pareto_front.csv` + `results/pareto.png` — the accuracy/speed/peak/energy trade-off.
- `results/facility_power.csv` + `.png` — scaled facility profile (Part ii PoC).

## Measurement caveat (kept honest, per the proposal)
tegrastats / INA3221 sample at ~ms, so this captures **peak power and thermal
transients** — not µs-scale di/dt. On-die nanosecond droop is out of scope.
