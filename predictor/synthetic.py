"""
predictor/synthetic.py — generate a realistic-sized config->waveform dataset so you
can build and validate the predictor (RQ2) and optimizer (RQ3) BEFORE the Orin data
is collected. Replace with data/waveforms.csv once real measurements exist.

The synthetic response surface encodes the known qualitative physics so the trained
predictor and the optimizer behave sensibly:
  * lower precision -> lower peak power, lower energy/inf (until memory-bound),
  * bigger batch    -> higher peak, higher throughput, higher energy/inf,
  * higher power mode-> higher peak ceiling and steeper thermal ramp,
  * LLMs            -> larger peak-to-average (prefill spike) and more transients.
These are trends, not ground truth — the real dataset overrides them.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

import config
from power.telemetry import _synth_trace
from power.waveform_features import extract_features


def generate(n_per_config: int = 3, seed: int = 0,
             spec: config.SweepSpec | None = None) -> pd.DataFrame:
    spec = spec or config.SweepSpec(
        models=config.MODELS,
        precisions=["FP16", "INT8", "FP8", "INT4"],
        batch_sizes=[1, 2, 4, 8, 16],
        power_modes=config.DEFAULT_POWER_MODES,
        modes=["inference", "train"],   # learn both regimes
    )
    rng = np.random.default_rng(seed)
    rows = []
    for cfg in spec.iter_configs():
        for _ in range(n_per_config):
            dur = 2.0 + 0.5 * np.log2(cfg.batch_size + 1)
            tr = _synth_trace(dur, 5.0, cfg)
            # add run-to-run jitter so the predictor sees variance
            tr.p_total = tr.p_total * (1 + rng.normal(0, 0.02))
            n_inf = 50
            f = extract_features(tr, n_inferences=n_inf)
            row = cfg.as_row()
            row.update(f.as_row())
            row["throughput_infps"] = n_inf / dur
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/synthetic_waveforms.csv", index=False)
    print(f"wrote data/synthetic_waveforms.csv  shape={df.shape}")
