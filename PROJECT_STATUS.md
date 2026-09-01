# Project Status & Walkthrough — Edge-Inference Power Thesis

**"Managing Power Behavior of ML Inference: From Edge Device to the Data Centers"**
Ardacan Yildiz · M.Sc., University of Akron · Advisor: Prof. Yilmaz Sozer
Status snapshot generated **September 1, 2026** · Hard deadline: **OhioLINK upload Nov 13, 2026**

---

## 1. TL;DR — where we are

You are **ahead of your own proposal timeline**. The proposal put "Characterize / Predict /
Choose" (Phases 2–3) in mid-September through mid-October. As of the last Orin run you already
have all three substantially done on **real hardware**:

- **540 real measured rows** are collected and committed — every single one from the Orin's
  INA3221 sensor (`trace_source == ina3221`), no synthetic/mock numbers in the results.
- The **RQ2 predictor** is trained and scoring well (energy R² = 0.995, peak R² = 0.979).
- The **RQ3 optimizer + Pareto** and the **full 13-figure set** are built from that real data.
- Two serious measurement bugs were found and fixed (rail double-count; missing `is_train`).

What is **not** done yet is the *breadth* the proposal advertises (INT8/INT4, YOLO, the Llama
LLM) and **RQ4 for real** (the facility scale-up currently runs on a placeholder trace, not a
measured one, and is not wired into the main pipeline). Those are the remaining Orin jobs.

**Bottom line:** the machinery is finished and validated on a narrow-but-real dataset. The
remaining work is *more measurement on the Orin* to widen coverage, one honest RQ4 pass, and
writing. You are in good shape against the Nov 13 deadline.

### Housekeeping done today
Cleaned the working folder from **1.5 GB → 275 MB**: removed an accidental nested duplicate
(`results/results/`, 429 MB), two regenerable ONNX export caches (~816 MB), all `__pycache__`
folders, and a stray temp file. Nothing unique was lost — the canonical `results/` is intact,
and the ONNX files rebuild automatically on the next Orin run.

---

## 2. What the experiment actually is (the science in one page)

A GPU running ML does **not** draw constant power — it draws a spiky *waveform*. The thesis
characterizes how your deployment choices reshape that waveform on a cheap edge GPU (Jetson
Orin Nano), then predicts it, optimizes under a power budget, and scales one device up toward
a data-center picture. Four research questions:

| RQ | Question | Status |
|----|----------|--------|
| **RQ1 — Characterize** | How do deployment settings (model, precision, batch, power mode, inference-vs-train) shape the measured power *waveform*? | ✅ Done on real data (540 rows) |
| **RQ2 — Predict** | Can a small model predict the waveform features from settings alone? | ✅ Trained, R² reported |
| **RQ3 — Optimize** | Given a power/energy ceiling, which config is best? | ✅ Pareto front + figures built |
| **RQ4 — Scale up** | Feed one Orin into a data-center model for facility-scale power? | ⚠️ Placeholder only — see §6 |
| *RQ2.5 — Per-primitive* | *(proposed)* isolate conv/depthwise/residual/matmul waveforms and reconstruct a whole network compositionally | 🔲 Idea, not built |

**The two "models" people confuse.** (1) The *inference models you benchmark* — ResNet-18,
ResNet-50, MobileNet-V3 (and later YOLO / Llama) — are run under load so their power is
measured. (2) The *predictor you actually train* (`predictor/train_predictor.py`) is the ML
core of RQ2: a small gradient-boosting model that learns **config → waveform features**.

**Where the real novelty lives** (from `NOVELTY.md`): almost every individual step has prior
art. The defensible wedge is the **target variable** — predicting *transient waveform shape*
(crest factor, dP/dt, thermal ramp), not the scalar watts/joules that prior work (NeuralPower,
nn-Meter, DeepEn) predicts — plus threading the whole pipeline through **one $250 device**.
Guard RQ4 carefully: Microsoft's 2026 "From Servers to Sites" (arXiv 2603.18383) already does
compositional trace generation server→site, so RQ4 must be framed as an *accessible proxy*, not
a new phenomenon.

---

## 3. The measurement, precisely

- **Device:** Jetson Orin Nano (Super), JetPack 6.2 / L4T r36.4.3, CUDA 12.6, PyTorch 2.8.
- **Sensor:** on-board **INA3221**, read from sysfs every **~5 ms**.
- **Total power = the `VDD_IN` input rail alone** — *not* the sum of VDD_IN + VDD_CPU_GPU_CV
  + VDD_SOC, because VDD_IN already **is** their sum (summing them double-counts ~2×, which is
  exactly the bug that once made peaks look like 23–69 W). After the fix, peaks are a plausible
  **7.7–22.1 W**.
- **Resolution caveat (documented, intentional):** at ~ms sampling you capture **peak power and
  thermal transients**, *not* µs-scale di/dt. On-die nanosecond droop is explicitly out of scope.

**The 540-row dataset design (RQ1):**
3 models × {FP16, FP32} × batch {1, 2, 4, 8} × power {15 W, 25 W, MAXN} × {inference, train} ×
3 repeats. Split evenly: 180 rows per model, 270 inference / 270 train.

Each row reduces one power-vs-time trace to scalar features:
`p_peak_w`, `p_avg_w`, `peak_to_avg` (crest factor), `dpdt_max_w_per_s`, `dpdt_p95_w_per_s`
(transient slope), `thermal_ramp_c_per_s`, `temp_max_c`, `energy_per_inf_j`, `energy_total_j`,
`throughput_infps`, plus `backend` and `trace_source`.

**A subtlety worth defending in the thesis:** the *sweep* feeds **dummy `torch.randn` tensors**
(power depends on the compute-graph shape, not pixel content — this makes runs reproducible and
removes dataloader noise), while the *genuine per-epoch training curves* use **real CIFAR-10**.
Both are legitimate and this is stated in `METHODS.md`.

---

## 4. File-by-file map — which analytical operation happens where

### Data flow (one line)
```
run_sweep.py ──> data/waveforms.csv ──┬─> predictor/train_predictor.py ─> results/predictor.pkl (+metrics.json)
 (RQ1 measure)     (the RQ1 dataset)   ├─> optimizer/optimize.py ────────> results/pareto_front.csv, surface_data.csv
                                       ├─> figures.py ───────────────────> results/figures/*.png
                                       └─> datacenter/scale_up.py ───────> facility_power.csv/.png  (RQ4, see §6)
train_capture.py ─> data/train_epochs.csv + data/train_traces/   (real per-epoch loss+power)
```

### Core scripts
| File | RQ | What it computes |
|------|----|------------------|
| `run_sweep.py` | RQ1 | The sweep driver. For each config: set nvpmodel power mode, load model, warm up, **log INA3221 while running N iters** (inference or train), extract the 7 waveform features, append a row to `data/waveforms.csv`, and dump the raw per-time trace to `results/traces/`. Per-config `try/except` — an OOM config is skipped and reported, never fatal. `NVPMODEL_IDS = {"MAXN":2, "25W":1, "15W":0}` (verified correct for **your** board). |
| `train_capture.py` | RQ1 | **Real epoch training** on CIFAR-10 (vision) — per-epoch loss + power + energy → `data/train_epochs.csv` and full traces to `data/train_traces/`. This is the honest loss-vs-power-over-time curve. LLM path is QLoRA-only. |
| `config.py` | — | Defines the `DeployConfig` object and the configuration space (the "deployment levers"). |

### Modules (the actual analytics)
| File | RQ | Operation |
|------|----|-----------|
| `power/telemetry.py` | RQ1 | INA3221 acquisition. Discovers rails, **de-duplicates by realpath**, selects the total `VDD_IN` rail, samples every ~5 ms. Prints `[power] rails=… total_rail=VDD_IN` on start. Falls back to a synthetic generator off-Jetson (tagged `mock`). **This is where the double-count bug was fixed — never "sum all rails" again.** |
| `power/waveform_features.py` | RQ1 | Reduces a raw trace to the 7 scalar features RQ2 predicts / RQ3 constrains. |
| `predictor/train_predictor.py` | **RQ2** | The ML core. **GradientBoostingRegressor, one per target** (multi-output). X = numeric config encoding (`precision_bits`, `is_int`, `batch_size`, `power_budget_w`, `is_llm`, **`is_train`**, schedule flag, model one-hot); Y = the 7 features. 80/20 split, reports **R² + MAE + feature importances**. `--backend mlp` swaps in a small PyTorch MLP. Outputs `predictor.pkl`, `predictor_metrics.json`. **The `is_train` bug (energy R² was −0.81) was fixed here — keep `mode` flowing.** |
| `predictor/synthetic.py` | RQ2 | Generates a synthetic dataset for laptop debugging *only* (never a result). |
| `optimizer/optimize.py` | **RQ3** | Scores every config, keeps the **non-dominated (Pareto) set** — minimize peak & energy, maximize throughput — under a peak-power budget. Outputs `pareto_front.csv`, `surface_data.csv`. |
| `datacenter/scale_up.py` | **RQ4** | Superposes N single-device traces with a `sync_fraction` knob (training = coherent/aligned spikes; inference = incoherent/averages out). Exports an OpenG2G-format trace. **Currently analytic placeholder — see §6.** |
| `figures.py` | RQ1/RQ3 | The **honest** figure module — loads only `trace_source == ina3221` rows. `grid_surface` plots a **measured** peak-power surface (every vertex a real measurement, no interpolation). Functions: `compare_peak`, `compare_energy`, `precision_compare`, `scatter_3d`, `grid_surface`, `train_per_epoch`, `power_over_time`. |

### Orchestration & export
| File | Role |
|------|------|
| `run_all.sh` | Fail-loud one-command pipeline **on the Orin**: archives any prior run into `_prev_<stamp>/`, then RQ1 sweep → real epoch training (3 models) → RQ2 predictor → RQ3 optimizer → MATLAB export → figures. Each stage prints `[OK]`/`[FAIL]`. **Note: does NOT run the RQ4 scale-up step — §6.** |
| `export_for_matlab.py` + `pareto_surface.m` | Export `surface_data.csv` and view the interactive surface of the real data in MATLAB. |

### Documentation (read order for anyone new)
`AGENTS.md` (authoritative onboarding) → `METHODS.md` (RQ1–RQ3 design) → `NOVELTY.md` (literature
audit / where you're actually novel) → `README.md` (quickstart) → `DEPLOY_ORIN.md` / `RUN_ON_ORIN.md`
(getting onto the board). `HANDOVER.md` is older and partly stale — `AGENTS.md` supersedes it.

---

## 5. The results you already have (real numbers)

**RQ2 predictor — held-out R²** (`results/predictor_metrics.json`):

| Target | R² | Reads as |
|--------|-----|----------|
| `energy_per_inf_j` | **0.995** | energy/iteration — predicted very well |
| `p_avg_w` | **0.980** | average power — well |
| `p_peak_w` | **0.979** | peak power — well |
| `thermal_ramp_c_per_s` | 0.69 | thermal ramp — harder |
| `dpdt_p95_w_per_s` | 0.68 | transient slope — harder |
| `dpdt_max_w_per_s` | 0.54 | peak di/dt — hardest |
| `peak_to_avg` | 0.52 | crest factor — hardest |

**This split is itself a finding, not a failure.** Static config predicts *magnitude*
(peak/avg/energy) cleanly, but predicts *transient shape* (dP/dt, crest) poorly because those
depend on thermal history and throttling. Top feature importances: `batch_size` (0.31),
`is_train` (0.17), `power_budget_w` (0.16), then model family. Lean into this in the write-up —
it's a clean, defensible result and it directly supports your novelty angle (transient shape is
the hard, interesting target).

**Architecture-fingerprint hypothesis, supported by the data:** depthwise-separable (MobileNet)
= memory-bound = *lower* power; dense conv + residual (ResNet) = compute-bound = *higher* power
but more efficient; depth (ResNet-50) multiplies *energy*, not peak.

Figures: 13 PNGs in `results/figures/` (peak/energy comparisons with error bars, FP32-vs-FP16,
3-D scatter, measured grid surface, per-epoch training power, power-over-time).

---

## 6. What still needs to happen on the Orin

These are the concrete measurement jobs left. All of them are "run on the board, `scp` results
back" — the analysis code is already written and validated.

### A. Broaden the RQ1 sweep to match the proposal's claims (highest value)
Your current dataset is **3 vision models × FP16/FP32 only**. The proposal advertises more.
Closing these gaps is what turns "narrow but real" into "as scoped":

1. **INT8 (and INT4 where supported)** — this is the *headline* precision story and it's
   currently missing. Real low-precision power needs a **TensorRT engine** (ONNX export is
   already done in `models/vision.py`; the `TODO` is the engine build). Without it, INT8 falls
   back to the FP path and isn't a true low-precision number.
2. **`batch_size = 16`** — one more point densifies the peak-power surface (OOM-risky on FP32;
   the sweep will skip-and-report rather than crash).
3. **YOLOv8n** — needs `ultralytics` on the board (`pip install --user ultralytics --no-deps`).
4. **Llama-3.2-1B** — the "small LLM" from the proposal. Needs `transformers`/`peft`; for a
   *true* INT4/INT8 decode power number, swap the HF model in `models/llm.py` for a
   TensorRT-LLM engine or a llama.cpp GGUF. Training = **LoRA/QLoRA only** (never full-weight on
   an 8 GB Nano).

### B. RQ4 for real — this is the biggest genuine gap
Right now the only `facility_power.*` output lives in the **old** `results_orin/` folder (Aug 27),
it was **not** regenerated in the current run, and `datacenter/scale_up.py` feeds the scaler a
**synthetic `_synth_trace` placeholder**, not a measured Orin trace. Also, `run_all.sh` does not
call the scale-up step at all. To make RQ4 defensible:

1. Feed **real measured *training* traces** (from `data/train_traces/`) into the superposition
   model instead of the synthetic placeholder.
2. **Validate the coherent-superposition assumption** against even 2–4 real synchronized devices
   before claiming a data-center prediction — `NOVELTY.md` flags this as your single most
   contestable claim.
3. Either wire up the real **OpenG2G** run (the CSV export is the intended input) or present the
   analytic scaler explicitly as a labeled proof-of-concept, and add the step to `run_all.sh`.

### C. Environment reminders for the board (don't relitigate — from `AGENTS.md`)
- **Never run `apt` / `apt update` on the Orin** (it crashed the board before). `pip3 install --user` only.
- Torch stack: `torch==2.8.0` + `torchvision==0.23.0` from `https://pypi.jetson-ai-lab.io/jp6/cu126`; pin `numpy==1.26.4`.
- **Only Orin INA3221 numbers are real.** Never present synthetic/mock numbers as results.
- Confirm `trace_source == ina3221` and `backend == torch-cuda-…` (not `stub`) on every run.
- Git runs **from Windows with OneDrive paused** (OneDrive fights `.git` locks) — not through the device shell.

---

## 7. Small things I noticed (low-priority tidy-ups)

- **`AGENTS.md` and `NOVELTY.md` are untracked in git** (uncommitted). They're your best docs —
  commit them (from Windows, OneDrive paused) so they're safe on origin/main.
- **`DEPLOY_ORIN.md` is partly stale**: it references `make_figures.py` (now archived — it's
  `figures.py`), suggests a `thesisenv` venv and an example `NVPMODEL_IDS` with MAXN=0, all of
  which conflict with the authoritative `AGENTS.md` / `RUN_ON_ORIN.md` (no venv; MAXN=2). Worth a
  quick reconciliation pass so future-you doesn't follow the wrong doc.
- **`results_orin/` (105 MB, first run)** and **`figures_current/` (5 figures)** are superseded by
  the canonical `results/` and the `results_orin_v2` backup. I left them (you chose the moderate
  cleanup), but they're safe to delete later — `results_orin/`'s only unique content is its RQ4
  `facility_power.*`, which you'll regenerate properly per §6B anyway.

---

## 8. Where we're headed — suggested path to Nov 13

1. **Now → mid-Sep:** broaden the sweep on the Orin — INT8 via TensorRT first (biggest payoff),
   then batch-16, then YOLO/Llama if time allows. Re-run `run_all.sh`, `scp` back.
2. **Mid-Sep → early-Oct:** re-train the predictor on the wider data; the INT8 rows are what let
   you actually answer the proposal's "does lower precision lower peak power *and* energy, and
   where's the crossover" question (RQ3's real payoff).
3. **Early → mid-Oct:** do RQ4 honestly (§6B) — real training traces + a 2–4 device superposition
   sanity check, framed as an accessible proxy.
4. **Mid-Oct → Nov 13:** writing (drafting can run in parallel from now). Defense early November,
   corrections, OhioLINK upload by the hard deadline.

You have the infrastructure and a validated real-data result already in hand — the remaining risk
is *board time* for the INT8/breadth measurements and one clean RQ4 pass, not method or code.
