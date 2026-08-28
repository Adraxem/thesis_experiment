# Experimental Design & Methodology
### Managing the Power Behavior of ML Inference — from Edge Device to Data Center

This document describes exactly what the experiment does: which models we run, in
which configurations, on what data, what we measure, and what predictive model we
train to forecast power. It maps one-to-one to the code (`run_sweep.py`,
`train_capture.py`, `predictor/`, `optimizer/`, `figures.py`).

---

## 0. Platform & how power is measured

- **Device:** NVIDIA Jetson Orin Nano (Super), JetPack 6.2 (L4T r36.4.3), CUDA 12.6, PyTorch 2.8.
- **Power sensor:** on-board **INA3221**, read from sysfs every **~5 ms**.
- **Total power** = the **VDD_IN** input rail alone. The INA3221 also exposes the
  component rails (VDD_CPU_GPU_CV, VDD_SOC); we do **not** add those to VDD_IN, because
  VDD_IN already *is* their sum — adding them double-counts power (~2×). Rails are also
  de-duplicated so the same channel is never counted twice.
- Every run yields a **power-vs-time waveform**, which we reduce to scalar features.

Sampling caveat (documented, intentional): at ~ms resolution we capture **peak power and
thermal transients**, not µs-scale di/dt.

---

## RQ1 — Characterize: build the config → waveform dataset

We sweep the deployment space. For each configuration we run the workload while the
INA3221 logs power, then extract features. This is the dataset that does not otherwise
exist on real hardware.

### Models (real, torchvision)
`resnet18`, `resnet50`, `mobilenet_v3_large`.
*(Future: `yolov8n` needs `ultralytics`; `llama-3.2-1b` needs transformers; these are
add-ons. True INT8/INT4 needs a TensorRT engine — future work.)*

### Configuration axes (the "deployment levers")
| Lever | Values |
|---|---|
| precision | FP16, FP32 |
| batch size | 1, 2, 4, 8 |
| power mode (`nvpmodel`) | 15 W, 25 W, MAXN_SUPER |
| workload mode | **inference** (forward only) and **training** (forward + backward + optimizer step) |
| repeats | 3 (for variance / error bars) |

≈ 3 models × 2 precisions × 4 batches × 3 power modes × 2 modes × 3 repeats ≈ **430 measured runs**.

### What "dataset" the workload uses
- **For power characterization (the sweep):** inputs are **dummy random tensors**
  (`torch.randn`). The GPU executes the *same* convolution/matmul kernels regardless of
  pixel content, so the power waveform is identical to real images. Using random tensors
  makes runs reproducible and removes dataloader noise. Accuracy is irrelevant here — we
  are measuring the *machine*, not training a classifier.
- **For genuine epoch training (per-epoch loss + power):** **CIFAR-10** (real images),
  8 epochs, capped steps/epoch. This gives the real loss-vs-power-over-time curve.
- **For large models (future):** LoRA/QLoRA on a small text corpus — backprop flows only
  through the adapters; the frozen base is never full-trained on a Nano.

### What we measure per run (the RQ1 feature columns)
- **Power:** `p_peak_w`, `p_avg_w`, `peak_to_avg` (crest factor)
- **Transients:** `dpdt_max_w_per_s`, `dpdt_p95_w_per_s` (how sharply power ramps — di/dt proxy)
- **Thermal:** `thermal_ramp_c_per_s`, `temp_max_c`
- **Energy:** `energy_per_inf_j` (energy per iteration), `energy_total_j`
- Plus `throughput_infps`, `backend`, `trace_source` (`ina3221`=real).

### RQ1 outputs
- `data/waveforms.csv` — one row per (config, repeat). **This is the RQ1 dataset.**
- `results/traces/<config>.csv` — the full power-vs-time waveform per run.
- `data/train_epochs.csv` + `data/train_traces/` — per-epoch loss+power and training traces.

---

## RQ2 — Predict: train a model to forecast the power outcomes

**Goal:** learn a mapping **config → power-waveform features**, so a new deployment's
power can be predicted without measuring it.

- **Training data:** `data/waveforms.csv` (the real RQ1 measurements — inference *and* training rows).
- **Inputs (X):** a numeric encoding of each config — `precision_bits`, `is_int`,
  `batch_size`, `power_budget_w`, `is_llm`, **`is_train`**, schedule flag, and a one-hot
  of the model family. (`is_train` matters: training and inference differ ~5× in energy.)
- **Targets (Y):** the 7 waveform features above.
- **Model:** a **Gradient Boosting Regressor** (scikit-learn), **one regressor per target**
  (multi-output). Chosen for robustness on small data, no GPU needed, and interpretable
  feature importances. *(An alternative `--backend mlp` trains a small PyTorch MLP — the
  "small learned model" named in the proposal.)*
- **Evaluation:** 80/20 train/test split; report **R² and MAE per target**, plus feature
  importances.
- **Interpretation of results:** power/energy targets (peak, avg, energy) predict well
  (R² ≈ 0.97–0.99); transient/thermal targets (dP/dt, thermal ramp) are hard to predict
  from static config alone because they depend on thermal history and throttling — this
  is itself a finding, not a failure.
- **Outputs:** `results/predictor.pkl`, `results/predictor_metrics.json`.

---

## RQ3 — Optimize: choose deployments under a power budget

Use the RQ2 predictor (or the measured rows directly) to select configurations that are
**Pareto-optimal** in **{peak power, energy per iteration, throughput}** subject to a
power budget (e.g. peak ≤ 50 W).

- **Method:** score every config → keep the **non-dominated set** (minimize peak & energy,
  maximize throughput) = the Pareto front.
- **Figures (measured, no interpolation fantasy):**
  - `grid_surface.png` — measured **peak-power surface over (power mode × batch)**; every
    vertex is a real measurement.
  - `scatter_3d.png` — measured points in (peak, energy, throughput).
  - `compare_peak.png` / `compare_energy.png` — model comparison with error bars.
  - `precision_compare.png` — FP32 vs FP16.
- **Outputs:** `results/pareto_front.csv`, `results/surface_data.csv` (+ `pareto_surface.m` for MATLAB).

---

## RQ4 — Scale-up: one device → a data center (in progress)

Superpose **N** single-device **training** traces with a synchronization/jitter knob
(`sync_fraction`) to model a facility. Training is barrier-synchronized, so per-device
power spikes align → coherent superposition; inference is incoherent and averages out.
The single Orin measures the per-device pattern; the model scales the coupling. A single
Nano cannot reproduce a multi-GPU spike — it is modeled from the measured per-device trace.

---

## How to reproduce (Orin)

```bash
cd ~/Desktop/thesis_experiment
bash run_all.sh          # scope editable at the top of the script
```

`run_all.sh` runs RQ1 (sweep + real epoch training) → RQ2 (predictor) → RQ3 (optimizer +
figures), archiving any previous run into `_prev_<timestamp>/` first so outputs are clean.
It is **fail-loud** (each stage prints OK/FAIL) and the sweep is **per-config resilient**
(an out-of-memory config is skipped and reported, never fatal).
