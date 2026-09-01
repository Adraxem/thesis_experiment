# AGENTS.md — agent onboarding for the edge-inference power thesis

**Read this first if you are an AI agent (or a future me) picking up this project.**
This supersedes the older `HANDOVER.md` (kept for history but partly stale — e.g. it still
describes synthetic PC data and a Q4 stub). Where the two disagree, **AGENTS.md wins.**

---

## 1. What this project is

M.Sc. thesis code for **Ardacan Yildiz**, University of Akron, advisor Prof. Yilmaz Sozer.
Title (proposal v4): *"Managing Power Behavior of ML Inference: From Edge Device to the
Data Centers."* Hard deadline: **OhioLINK upload Nov 13, 2026.**

One line: on a **Jetson Orin Nano (Super)**, characterize how deployment choices
(model, precision, batch, power mode, and **inference vs training**) shape the *power
waveform* (peak, transient sharpness, thermal ramp, energy), then **predict** it,
**optimize** under a power budget, and **scale** one device up to a data center.

Four research questions: **RQ1** characterize · **RQ2** predict · **RQ3** optimize ·
**RQ4** facility scale-up. (A proposed **RQ2.5** = per-primitive/per-layer profiling; see §7.)
Full methodology is in `METHODS.md` — read it for the RQ1→RQ3 design in detail.

Repo: https://github.com/Adraxem/thesis_experiment (MIT). The authoritative proposal is
`Yildiz_Thesis_Proposal_v4.docx` in the sibling `thesis/` folder (v1–v3 are OLD, ignore).

---

## 2. Current status (Aug 2026) — REAL DATA IS IN

The pipeline has been run for real on the Orin. **540 measured rows** are committed.

- `data/waveforms.csv` — **540 rows, every one `trace_source == ina3221` (real sensor).**
  Axes: 3 models × {FP16,FP32} × batch {1,2,4,8,16} × power {15W,25W,MAXN} ×
  {inference,train} × 3 repeats. Measured peak-power range **7.7–22.1 W**.
- Predictor (RQ2), optimizer + Pareto (RQ3), and the full figure set are all built from
  this real data. 13 figures in `results/figures/`.
- Last commit at HEAD: *"540-row Orin dataset (3 models, FP16/FP32, inference+train) +
  predictor/optimizer + full figure set + METHODS"* — already pushed to origin/main.

### RQ2 predictor results (from `results/predictor_metrics.json`, held-out R²)
| Target | R² | Reads as |
|---|---|---|
| `p_avg_w` | **0.980** | avg power — predicted well |
| `p_peak_w` | **0.979** | peak power — predicted well |
| `energy_per_inf_j` | **0.995** | energy/iter — predicted very well |
| `thermal_ramp_c_per_s` | 0.69 | thermal — harder |
| `dpdt_p95_w_per_s` | 0.68 | transient slope — harder |
| `dpdt_max_w_per_s` | 0.54 | peak di/dt — hardest |
| `peak_to_avg` | 0.52 | crest factor — hardest |

**This split is itself a finding, not a failure:** static config predicts *magnitude*
(peak/avg/energy) cleanly; it predicts *transient shape* (dP/dt, crest) poorly because
those depend on thermal history and throttling. Top feature importances: `batch_size`
(0.31), `is_train` (0.17), `power_budget_w` (0.16), then model family.

---

## 3. Two scientific bugs that were found and fixed — do NOT reintroduce them

These were the two biggest corrections in the project. If numbers ever look wrong, check
these first.

1. **INA3221 rail double-count (was inflating power ~2×, peaks looked like 23–69 W).**
   `power/telemetry.py` must use the **total input rail `VDD_IN` ALONE**, never the sum of
   `VDD_IN + VDD_CPU_GPU_CV + VDD_SOC` — VDD_IN already *is* their sum. The fix:
   `_discover_rails()` de-duplicates by realpath; `_select_total_rail()` picks the total
   rail via `_TOTAL_RAIL_RE`; `_sample_loop` uses that rail alone if present, else sums
   only leaf rails. On start it prints `[power] rails=... total_rail=VDD_IN`. After the fix,
   peaks are a plausible **7.7–22.1 W**. **Never "sum all rails" again.**
2. **Predictor `is_train` was always 0** (energy R² was **−0.81**, nonsense).
   `predictor/train_predictor.py::_feature_frame` was building `DeployConfig` without
   `mode=`, so training vs inference was invisible to the model. Fix: pass
   `mode=r.get("mode", "inference")`. Energy R² went to **+0.995**. Keep `mode` flowing.

---

## 4. File map (what runs what)

- `run_sweep.py` — RQ1 sweep. `NVPMODEL_IDS = {"MAXN":2, "25W":1, "15W":0}` (correct for
  THIS board). Per-config `try/except` (an OOM config is skipped + reported, never fatal).
  Saves `data/waveforms.csv` after every config, and per-time traces to
  `results/traces/<tag>_r<rep>.csv` with `--save-traces`.
- `train_capture.py` — REAL epoch training (CIFAR-10), per-epoch loss + power → `data/train_epochs.csv` and `data/train_traces/`.
- `power/telemetry.py` — INA3221 measurement (see §3). `power/waveform_features.py` reduces
  a trace to the 7 scalar features RQ2 predicts and RQ3 constrains.
- `predictor/train_predictor.py` — RQ2. GradientBoostingRegressor, one per target
  (multi-output). Outputs `results/predictor.pkl`, `results/predictor_metrics.json`.
- `optimizer/` — RQ3 Pareto under a peak-power budget → `results/pareto_front.csv`,
  `results/surface_data.csv`.
- `figures.py` — the **honest** figure module. Loads only `trace_source=="ina3221"` rows.
  Functions: `compare_peak`, `compare_energy`, `precision_compare`, `scatter_3d`,
  `grid_surface` (measured grid — **every vertex is a real measurement, no interpolation**),
  `train_per_epoch`, `power_over_time`. `main()` wraps each figure in try/except.
  (Header patched for the Orin's dual-matplotlib: prepends user site-packages to `sys.path`.)
- `run_all.sh` — fail-loud orchestrator. `step()` prints `[OK]`/`[FAIL]`; archives any prior
  run into `_prev_<STAMP>/` first. Scope editable at the top.
- `datacenter/` — RQ4 scale-up (superpose N training traces with a `sync_fraction` knob).
- Helper scripts staged on the Orin: `orin_setup.sh`, `orin_probe.sh`, `orin_sweep.sh`,
  `orin_analyze.sh`.

---

## 5. Hard constraints — do NOT relitigate these

1. **NEVER run `apt` / `apt update` on the Orin.** It crashed the board before; the user
   forbade it. Install Python deps with `pip3 install --user` only (no venv — `python3-venv`
   is absent and installing it needs apt).
2. **Torch stack on the Orin:** `torch==2.8.0` + `torchvision==0.23.0` from
   `https://pypi.jetson-ai-lab.io/jp6/cu126` (JetPack 6.2 / L4T r36.4.3 / CUDA 12.6).
   Newer torch (2.11) fails with `libcudss.so.0`. Pin `numpy==1.26.4` (torch built for
   numpy 1.x). Use `python3 -m pip ...` so the right interpreter is hit (system pandas can
   shadow the `--user` one).
3. **Only Orin INA3221 numbers are real.** Any PC/synthetic power is FAKE (`trace_source`
   `mock`). Do not present synthetic numbers as results. No made-up plots (this killed the
   old cubic-interpolated `pareto_3d.png` with its fantasy 40 W surface).
4. **Large-model training = LoRA/QLoRA only** — never full-weight train a large model on an
   8 GB Nano. Backprop through adapters only.
5. **Dummy random tensors vs CIFAR:** the power *sweep* uses `torch.randn` dummy tensors
   (power is a function of the compute-graph shape, not pixel values — reproducible, no
   dataloader noise). Real **epoch training curves** use CIFAR-10. Both are legitimate; see
   `METHODS.md` §RQ1.

---

## 6. Machines, deploy, and the git/OneDrive rule

- **Cloud container** (this agent's sandbox): has `Bash`, web, skills. No GitHub creds, and
  **cannot reach the Orin.**
- **User's desktop `kal-el-laptop`** (win32): reached via `device_bash`; the repo is the
  mounted folder `C:\Users\ardac\OneDrive\Desktop\thesis_experiment`.
- **The Orin `ardacan-desktop`** (IP **10.18.29.68**): reached ONLY from the user's own
  terminal via `scp`/`ssh`. Agents do NOT have a direct channel to it — hand the user
  copy-paste commands, they run them and `scp` results back.
- **Deploy loop:** edit code on desktop → user `scp`s scripts to the Orin → user runs them →
  user `scp`s `results/`, `waveforms*.csv`, `train_epochs*.csv` back to the desktop.
- **GIT/OneDrive:** the repo lives in OneDrive, which fights `.git` locks
  (`index.lock`/`HEAD.lock` "Operation not permitted" from `device_bash`; pushes rejected
  non-fast-forward). **All git operations run from Windows with OneDrive paused:**
  pause OneDrive → `Remove-Item .git\*.lock` → `git add -A` → `git commit` → `git push`.
  Do not attempt `git commit`/`push` through `device_bash`. Writing plain files (like this
  one) into the folder is fine.

---

## 7. Open ideas / next steps

- **RQ2.5 — per-primitive profiling (proposed, user evaluating).** `primitive_sweep.py`
  isolating conv / depthwise-separable / residual / matmul blocks, plus a leave-one-model-out
  *compositional* predictor that reconstructs a whole-network waveform from primitives.
  **Honest risk to state up front:** isolated-layer power ≠ in-network fused power (kernel
  fusion, cache residency). That "fusion residual" is itself a finding, not a bug.
- **Architecture-fingerprint hypothesis (supported by the data):** depthwise-separable
  (MobileNet) = memory-bound = *lower* power; dense conv + residual (ResNet) = compute-bound
  = *higher* power but more efficient; depth (ResNet-50) multiplies *energy*, not peak.
- **RQ4 for real:** feed measured *training* traces into the superposition model; validate
  the coherent-superposition assumption against even 2–4 real synchronized devices before
  claiming a datacenter prediction (see `NOVELTY.md` — this is the most contestable claim).
- Sustained-inference `--serve-seconds` mode for a long-horizon inference power trace (offered, not built).
- TensorRT INT8/INT4 engine for true low-precision peak power (future work; needs the engine built).

---

## 8. Briefing a new agent — paste this

> "Read `AGENTS.md` then `METHODS.md` in the repo root. Real data (540 rows) is in
> `data/waveforms.csv`, all from the Orin's INA3221. Don't reintroduce the two fixed bugs
> (rail double-count in telemetry; missing `is_train` in the predictor). Never run `apt` on
> the Orin. Only Orin INA3221 numbers are real — no synthetic results. Large-model training
> is LoRA/QLoRA only. Git runs from Windows with OneDrive paused. I'm working on <X>."
