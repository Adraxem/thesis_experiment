# HANDOVER — edge-inference power thesis (agent + human context)

Read this first if you're a new agent (or a future me) picking up this project.

## What this is
M.Sc. thesis code for **Ardacan Yildiz**, University of Akron, advisor Prof. Yilmaz Sozer.
Title (proposal v4): *"Managing Power Behavior of ML Inference: From Edge Device to the
Data Centers."* Hard deadline: **OhioLINK upload Nov 13, 2026.**
The authoritative proposal lives in the sibling `thesis/` folder (`Yildiz_Thesis_Proposal_v4.docx`) — v4 is current; v1–v3 and the prospectus are OLD, ignore them.

One-line thesis: on a **Jetson Orin Nano**, characterize how deployment choices
(precision, batch, power mode, model, and **inference vs training**) shape the *power
waveform* (peak, transient sharpness, thermal ramp, energy) — then predict it, optimize
under a power budget, and scale one device up to a data center.

Four research questions: Q1 characterize · Q2 predict · Q3 optimize · Q4 facility scale-up.

## Current status (Aug 2026)
- Full pipeline scaffolded and RUNS end-to-end on a PC with **synthetic** power, and on
  the Orin with **real** INA3221 power. Same code, auto-switches on `platform.machine()`.
- Inference AND on-device training are both captured.
- Predictor (Q2) trains; optimizer + Pareto (Q3) work; scale-up (Q4) is a STUB.
- Pushed to GitHub: https://github.com/Adraxem/thesis_experiment (MIT).

## Decisions already made — do NOT relitigate these
1. **PC power numbers are FAKE.** `power/telemetry.py::_synth_trace` invents them so the
   code runs pre-hardware. Real numbers ONLY come from the Orin (INA3221, see below).
   The `trace_source` column says `mock` (fake) vs `ina3221` (real).
2. **The precision→power assumption is NOT hardcoded in the analysis.** It lives only in
   the synthetic data generator. Measurement/features/predictor/optimizer assume nothing;
   they report whatever the sensor gives. Finding where the assumption breaks is a Q1
   result, not a bug.
3. **Two distinct meanings of "training":** (a) training real workload models (ResNet on
   CIFAR, Llama via QLoRA) on the Orin and capturing their power — this is FIRST-CLASS
   data and the building block for Q4; (b) training the predictor — the Q2 ML output.
   Both matter; don't conflate or dismiss (a).
3b. **Large-model training = LoRA/QLoRA only.** Never full-weight train a large model on
   a Nano. Backprop runs through adapters only. Training power is stationary, so short
   runs still characterize the waveform.
4. **Scale-up model (Q4):** superpose N copies of a single-device **training** trace with
   a synchronization/jitter knob (`datacenter/scale_up.py::scale_facility`, `sync_fraction`).
   Training is barrier-synchronized so coherent superposition is justified; inference is
   incoherent and averages out. The single Orin measures the per-device pattern; the
   scale-up MODELS the coupling. A single Orin CANNOT reproduce the multi-GPU spike.
5. Optimizer/Pareto operate on **inference** configs (deployment = serving).

## How power is measured (the core mechanism)
`power/telemetry.py`: line 30 `IS_JETSON`; on the Orin `_discover_rails()` (l.56) finds the
INA3221 sysfs files, `_sample_loop()` (l.181) samples every few ms, l.190 computes
`mW = mA*mV/1000`. `power/waveform_features.py` turns a trace into the scalar features
(peak, dP/dt, thermal ramp, energy) that Q2 predicts and Q3 constrains.

## File map (see README.md for detail)
config.py (config space) · run_sweep.py (Q1 sweep) · train_capture.py (REAL epoch training
+ per-epoch/per-time power) · power/ (measurement) · models/ (ResNet/MobileNet/YOLO + Llama)
· predictor/ (Q2 + synthetic dummy data) · optimizer/ (Q3) · datacenter/ (Q4 stub) ·
make_figures.py / pareto_3d.py / pareto_surface.m / export_for_matlab.py (figures) ·
run_all.bat (PC one-click) · DEPLOY_ORIN.md (deployment).

## Data artifacts
- data/synthetic_waveforms.csv — DUMMY (2400 rows), for building the predictor pre-hardware.
- data/waveforms.csv — the REAL config→waveform dataset once run on the Orin (Q1 output).
- data/train_epochs.csv + data/train_traces/ — REAL training power, per-epoch and per-time.
- results/ — trained predictor, Pareto, figures, MATLAB surface data.

## Open TODOs / next steps
- [ ] On Orin: verify `nvpmodel -q` IDs and fix `NVPMODEL_IDS` in run_sweep.py.
- [ ] models/vision.py: build the TensorRT engine (ONNX export already done) for true
      INT8/FP8 peak power.
- [ ] models/llm.py: swap HF fp16 for TensorRT-LLM or llama.cpp GGUF for real INT4 decode.
- [ ] Collect the real dataset on the Orin; retrain predictor with --data data/waveforms.csv
      (its R^2 is the actual Q2 result; the synthetic ~0.99 is meaningless).
- [ ] Build Q4 for real: wire scale_up to OpenG2G (arXiv:2605.05519), feed TRAINING traces.
- [ ] Add fig6: per-epoch training curve (loss + power + temp) from data/train_epochs.csv.

## Gotchas
- Python: use 3.12 (3.13 ok). 3.6/3.7 will NOT work (dataclasses/libs).
- numpy 2.x removed np.trapz (already handled in waveform_features.py).
- Repo lives in OneDrive — OneDrive can fight git's .git folder; move out if corruption.
- Llama-3.2-1B is gated on HF — set HF_TOKEN or use a local path.
- First CIFAR training run downloads ~170 MB.
- On Orin, use NVIDIA JetPack wheels for torch/torchvision; TensorRT/pycuda come with JetPack.

## Briefing a new agent — paste this
"Read HANDOVER.md and README.md in the repo root. The current thesis is proposal v4 in
the sibling thesis/ folder. I'm working on <X>. Don't treat PC/synthetic power as real
results; real data comes from the Orin. Ask before changing decisions listed in HANDOVER."
