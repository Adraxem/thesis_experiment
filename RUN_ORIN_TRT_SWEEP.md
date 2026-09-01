# Lab runbook — wide TensorRT precision sweep (FP32 → FP16 → INT8 → INT4)

Goal: collect a broad **real** dataset across precisions and batch sizes on the Orin, push
to GitHub, then **stop and inspect the 3D/broad-data views yourself** before RQ3/RQ4.

New code added for this (already on the desktop, needs pushing): `models/trt_engine.py`,
a rewritten `models/vision.py` (real TensorRT path + guards), `run_sweep.py --engine trt`,
and `figures.py --data/--out-dir`. TensorRT is **inference-only**, so this sweep is
`--modes inference`; training power stays on the torch path (separate optional pass).

> **Honesty guard baked in:** INT8/INT4 only run through a real `trtexec`-built engine.
> If you ever run them without `--engine trt`, the sweep **refuses** (it will not measure
> FP32 and mislabel it as INT8). If a precision can't build on your board, that config is
> **skipped and reported**, never faked.

---

## Step 0 — get the new code onto the Orin

**On the Windows desktop (PowerShell), with OneDrive PAUSED** (tray → OneDrive → Pause 2h):
```powershell
cd C:\Users\ardac\OneDrive\Desktop\thesis_experiment
Remove-Item .git\*.lock -ErrorAction SilentlyContinue
git add -A
git commit -m "Real TensorRT INT8/INT4 sweep path (trtexec) + honesty guards; figures --data; docs"
git push
```
**On the Orin:**
```bash
cd ~/Desktop/thesis_experiment          # wherever you cloned it
git fetch origin && git reset --hard origin/main
```

## Step 1 — sanity: trtexec + CUDA present
```bash
ls -l /usr/src/tensorrt/bin/trtexec                      # must exist (ships with JetPack)
python3 -c "import torch; print('cuda', torch.cuda.is_available())"   # must print True
```
If `trtexec` isn't at that path: `find / -name trtexec 2>/dev/null` and tell me — the helper
also checks PATH and /usr/local/tensorrt.

## Step 2 — PROBE (prove INT8/INT4 are real before the big run)
One model, one batch, all four precisions, ~2 min:
```bash
sudo -v
python3 run_sweep.py --engine trt \
  --models resnet18 --precisions FP32 FP16 INT8 INT4 \
  --batch-sizes 1 --power-modes MAXN --modes inference \
  --iters 100 --repeats 1 --out data/probe_trt.csv

python3 - <<'PY'
import pandas as pd
d=pd.read_csv("data/probe_trt.csv")
print(d[["precision","backend","trace_source","p_peak_w","p_avg_w","energy_per_inf_j","throughput_infps"]].to_string(index=False))
PY
```
**What to confirm:**
- `backend` reads `trt-fp32`, `trt-fp16`, `trt-int8` (and `trt-int4` if your board supports it).
- `trace_source` is `ina3221` on every row.
- INT8 shows **lower peak/energy or higher throughput** than FP16/FP32 — that's the real quantized kernels.
- If the **INT4 row is missing**, it was skipped; the sweep prints the reason. On Ampere/Orin,
  plain-ONNX INT4 usually needs NVIDIA ModelOpt Q/DQ — tell me and I'll wire that route. It's not a fake, just not supported yet.

## Step 3 — the wide sweep (this is the dataset)
Keep the existing 540-row set safe, write the new one to its own file:
```bash
cp data/waveforms.csv data/waveforms_torch540_backup.csv     # preserve the old inference+train set

sudo -v
sudo jetson_clocks                                            # optional: steadier clocks
python3 run_sweep.py --engine trt \
  --models resnet18 resnet50 mobilenet_v3_large \
  --precisions FP32 FP16 INT8 INT4 \
  --batch-sizes 1 2 4 8 16 \
  --power-modes 15W 25W MAXN \
  --modes inference \
  --iters 200 --repeats 3 --period-ms 5 \
  --out data/waveforms_trt.csv
```
- Precisions run **in the order given** (FP32 → FP16 → INT8 → INT4), one by one, per model.
- Engines build **once** per (model, batch, precision) and cache in `results/engines/`; repeats and
  power modes reuse them, so only the first pass pays the build cost.
- OOM configs (e.g. resnet50 @ batch16 FP32) **skip and report** — never fatal.
- The CSV is re-saved after every config, so a crash mid-run keeps finished rows. Expect ~1–3 h.

**Optional — training power (torch path, keeps the `is_train` regime for RQ2 later):**
```bash
python3 run_sweep.py --engine torch \
  --models resnet18 resnet50 mobilenet_v3_large \
  --precisions FP16 FP32 --batch-sizes 1 2 4 8 --power-modes 15W 25W MAXN \
  --modes train --iters 200 --repeats 3 --out data/waveforms_train.csv
```

## Step 4 — the 3D / broad-data views to inspect yourself
Write them to a separate folder so the old figures stay intact:
```bash
python3 figures.py --data data/waveforms_trt.csv --out-dir results/figures_trt
python3 export_for_matlab.py --data data/waveforms_trt.csv --out results/surface_data_trt.csv
```
Look at (in `results/figures_trt/`):
- **`precision_compare.png`** — peak power vs batch, one line per precision. *The* precision-effect view (FP32/FP16/INT8/INT4).
- **`grid_surface.png`** — measured peak-power **surface** over (power budget × batch), per model. Every vertex is a real measurement.
- **`scatter_3d.png`** — every deployment point in (peak power, energy/iter, throughput).
- **`compare_peak.png` / `compare_energy.png`** — model comparison with error bars.

Interactive rotatable MATLAB surface (speed vs peak-power vs energy, Pareto points in red):
```bash
cp results/surface_data_trt.csv results/surface_data.csv     # pareto_surface.m reads this path
```
then in MATLAB, from the repo folder: `>> pareto_surface`

## Step 5 — push everything to GitHub
The Orin is a plain Linux box (no OneDrive lock), so git works normally there:
```bash
git add -A
git commit -m "Wide TensorRT precision sweep FP32/FP16/INT8/INT4 (3 vision models, batch 1-16) + 3D figures"
git push
```
Then on the desktop (OneDrive paused): `git pull`.
(`results/engines/`, `results/traces/`, and `*.engine`/`*_fp32.onnx` are gitignored — the CSVs
and `results/figures_trt/` PNGs are what get committed.)

---

### Then STOP and look
That's the halt point. Inspect the four figures + the MATLAB surface. Bring me back anything
that looks off — especially whether INT8 actually lowers peak/energy where you'd expect, and
whether INT4 built at all — and we'll decide RQ3/RQ4 from what the broad data actually shows.
