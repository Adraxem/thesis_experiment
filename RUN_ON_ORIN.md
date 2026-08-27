# Running the pipeline on the Jetson Orin Nano (git-based, no USB)

## First time only (fresh board) — JetPack 6.2 / L4T r36.4.3
```bash
pip3 install --user pandas scikit-learn matplotlib scipy
pip3 install --user "numpy==1.26.4"
pip3 install --user "torch==2.8.0" "torchvision==0.23.0" \
    --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
python3 -c "import torch; print(torch.cuda.is_available())"   # must print True
```

## Every run
```bash
cd ~/Desktop/thesis_experiment            # wherever you cloned it
git fetch origin && git reset --hard origin/main   # take latest code, drop local tweaks
bash run_all.sh
```
`run_all.sh` runs the whole thing: RQ1 sweep (inference + training, real per-time
traces) -> per-epoch training power -> RQ2 predictor -> RQ3 optimizer -> 3D Pareto
-> every figure.

## Outputs
- `data/waveforms.csv`      RQ1 dataset (config -> waveform features)
- `results/traces/`         per-time power waveforms (power over time)
- `data/train_epochs.csv`   per-epoch loss + power
- `results/figures/fig1..fig7.png`, `results/pareto_3d.png`

## Get results back to your PC (no USB)
```bash
git add -A && git commit -m "orin run" && git push
```

## Sanity checks (the fixes)
- Peak power should be ~5-40 W, NOT 60-70 W. run_sweep prints `[power] rails=... total_rail=...`
  — if it can't find a VDD_IN total rail and sums leaves, that's fine; if peaks look ~2x high,
  paste that rails line.
- `trace_source` must be `ina3221` (real) and `backend` `torch-cuda-...` (not `stub`).
