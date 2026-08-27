"""
train_capture.py — REAL on-device training with power captured per-time AND per-epoch.

This is the training-power data collection (Q1, training regime; the building block
for the Q4 scale-up). Unlike run_sweep.py's quick step-loop, this does genuine
epoch-based training on a real dataset, logs the INA3221 power waveform continuously,
and writes a per-epoch summary (loss + power + energy + temperature each epoch).

Models:
  * vision (resnet18/50, mobilenet_v3_large): trained on CIFAR-10 (real backprop, full
    weights). Small models -> a few epochs is minutes on the Orin.
  * llama3.2-1b: QLoRA fine-tune on a small text corpus. Backprop runs ONLY through the
    LoRA adapters (the frozen base is never updated) — the only feasible way to "train"
    a 1B model on a Nano. Full-weight training of a large model is intentionally NOT done.

Time is under your control: --epochs and --max-steps-per-epoch bound it. Training power
is a stationary signal (every step ~ the same draw), so even a short run characterizes
the waveform; more epochs mainly give you the loss-vs-power-over-time curve.

Off the Orin (no torch/CUDA) it runs a MOCK epoch loop with synthetic power so you can
see the per-epoch / per-time output format on your PC. Real numbers require the Orin.

Outputs:
  data/train_epochs.csv            (appended)  one row per (run, epoch): loss + power feats
  data/train_traces/<tag>.csv                  the full per-time trace (t, power_w, temp_c, epoch)
  results/checkpoints/<tag>.*                   trained weights / LoRA adapters (--save-ckpt)

Examples:
  python train_capture.py --model resnet18 --epochs 3 --batch-size 64 --power-mode MAXN
  python train_capture.py --model llama3.2-1b --epochs 1 --max-steps-per-epoch 200 --save-ckpt
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import pandas as pd

import config
from power.telemetry import PowerLogger, PowerTrace, IS_JETSON
from power.waveform_features import extract_features
from models.base import _have

_TORCH = _have("torch")


# --------------------------------------------------------------------------- helpers
def _slice_trace(tr: PowerTrace, t0: float, t1: float) -> PowerTrace:
    """Extract the samples of a PowerTrace between relative times t0..t1 (seconds)."""
    m = (tr.t >= t0) & (tr.t <= t1)
    if m.sum() < 2:                       # fallback: keep at least the whole thing
        m = np.ones_like(tr.t, dtype=bool)
    sub = PowerTrace(t=tr.t[m] - tr.t[m][0], p_total=tr.p_total[m],
                     rails={k: v[m] for k, v in tr.rails.items()},
                     temp=(tr.temp[m] if tr.temp is not None else None),
                     period_ms=tr.period_ms, source=tr.source)
    return sub


def _epoch_index_for_samples(tr: PowerTrace, epoch_bounds):
    """Label each per-time sample with the epoch it falls in (for the trace CSV)."""
    idx = np.full(len(tr.t), -1, dtype=int)
    for e, (a, b) in enumerate(epoch_bounds):
        idx[(tr.t >= a) & (tr.t <= b)] = e
    return idx


# --------------------------------------------------------------------------- datasets
def _cifar_loader(batch_size, imgsz=224, root="data/datasets"):
    import torch
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(imgsz), T.ToTensor(),
                    T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    ds = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                       num_workers=2, pin_memory=True)


def _ensure_corpus(path="data/datasets/mini_corpus.txt"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(("Edge GPUs draw power in sharp bursts. Training synchronizes many "
                      "devices so their power spikes align. Quantization lowers peak power "
                      "but can raise total energy. The Jetson Orin measures power with an "
                      "on-board INA3221 sensor. ") * 200)
    return path


# --------------------------------------------------------------------------- vision train
def train_vision(cfg, epochs, max_steps, imgsz, save_ckpt, logger_period, out_prefix):
    import torch
    import torchvision.models as M
    device = "cuda" if torch.cuda.is_available() else "cpu"
    builder = {"resnet18": M.resnet18, "resnet50": M.resnet50,
               "mobilenet_v3_large": M.mobilenet_v3_large}[cfg.model]
    net = builder(weights=None, num_classes=10).to(device).train()
    if cfg.precision == "FP16" and device == "cuda":
        net = net.half()
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    lossf = torch.nn.CrossEntropyLoss()
    loader = _cifar_loader(cfg.batch_size, imgsz)

    def run_epoch():
        total, n, t0 = 0.0, 0, time.perf_counter()
        for step, (x, y) in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            if cfg.precision == "FP16" and device == "cuda":
                x = x.half()
            opt.zero_grad()
            out = net(x)
            loss = lossf(out.float(), y)
            loss.backward()
            opt.step()
            total += float(loss.item()); n += 1
        if device == "cuda":
            torch.cuda.synchronize()
        return total / max(n, 1), n, time.perf_counter() - t0

    ckpt = None
    if save_ckpt:
        os.makedirs("results/checkpoints", exist_ok=True)
        ckpt = f"results/checkpoints/{out_prefix}.pt"
    return _drive(cfg, epochs, run_epoch, logger_period, out_prefix,
                  save_fn=(lambda: torch.save(net.state_dict(), ckpt)) if ckpt else None,
                  backend=f"torch-{device}-{cfg.precision}")


# --------------------------------------------------------------------------- llm train (QLoRA)
def train_llm(cfg, epochs, max_steps, save_ckpt, logger_period, out_prefix):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from models.llm import MODEL_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)

    from peft import LoraConfig, get_peft_model
    lc = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    net = get_peft_model(base, lc).train()         # ONLY adapters get gradients
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)

    # tokenize the corpus into fixed blocks
    text = open(_ensure_corpus()).read()
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    L = cfg.seq_len
    blocks = [ids[i:i + L] for i in range(0, len(ids) - L, L)]

    def run_epoch():
        total, n, t0 = 0.0, 0, time.perf_counter()
        for step in range(0, len(blocks), cfg.batch_size):
            if max_steps and n >= max_steps:
                break
            batch = blocks[step:step + cfg.batch_size]
            if len(batch) < 1:
                break
            inp = torch.stack([b for b in batch if len(b) == L]).to(device)
            if inp.numel() == 0:
                continue
            opt.zero_grad()
            out = net(input_ids=inp, labels=inp)
            out.loss.backward()
            opt.step()
            total += float(out.loss.item()); n += 1
        if device == "cuda":
            torch.cuda.synchronize()
        return total / max(n, 1), n, time.perf_counter() - t0

    save_fn = None
    if save_ckpt:
        os.makedirs("results/checkpoints", exist_ok=True)
        save_fn = lambda: net.save_pretrained(f"results/checkpoints/{out_prefix}_lora")
    return _drive(cfg, epochs, run_epoch, logger_period, out_prefix,
                  save_fn=save_fn, backend=f"transformers-{device}-qlora")


# --------------------------------------------------------------------------- mock (PC)
def train_mock(cfg, epochs, logger_period, out_prefix):
    def run_epoch():
        t0 = time.perf_counter()
        time.sleep(0.4)                              # pretend an epoch of work
        loss = 2.3 * np.exp(-0.3 * run_epoch.e) + 0.1 * np.random.rand()
        run_epoch.e += 1
        return float(loss), 20, time.perf_counter() - t0
    run_epoch.e = 0
    return _drive(cfg, epochs, run_epoch, logger_period, out_prefix,
                  save_fn=None, backend="stub-train")


# --------------------------------------------------------------------------- the driver
def _drive(cfg, epochs, run_epoch, logger_period, out_prefix, save_fn, backend):
    """Common loop: log power continuously, run each epoch, slice power per epoch."""
    os.makedirs("data/train_traces", exist_ok=True)
    epoch_rows, bounds, losses, steps_list = [], [], [], []
    print(f"[train] {cfg.tag()}  backend={backend}  epochs={epochs}  jetson={IS_JETSON}")
    logger = PowerLogger(period_ms=logger_period, cfg=cfg)
    logger.start()
    t_start = logger._t0
    for e in range(epochs):
        a = time.perf_counter() - t_start
        loss, steps, dur = run_epoch()
        b = time.perf_counter() - t_start
        bounds.append((a, b)); losses.append(loss); steps_list.append(steps)
        print(f"  epoch {e+1}/{epochs}  loss={loss:.4f}  steps={steps}  time={dur:.1f}s")
    logger.stop()
    tr = logger.trace()

    # per-epoch power features (slice the continuous trace) + the epoch's loss
    for e, (a, b) in enumerate(bounds):
        sub = _slice_trace(tr, a, b)
        f = extract_features(sub, n_inferences=1)
        row = {"model": cfg.model, "precision": cfg.precision, "mode": "train",
               "batch_size": cfg.batch_size, "power_mode": cfg.power_mode,
               "epoch": e, "loss": round(losses[e], 5), "steps": steps_list[e],
               "epoch_time_s": round(b - a, 3),
               "p_avg_w": round(f.p_avg_w, 3), "p_peak_w": round(f.p_peak_w, 3),
               "energy_j": round(f.energy_total_j, 3),
               "thermal_ramp_c_per_s": round(f.thermal_ramp_c_per_s, 4),
               "temp_max_c": round(f.temp_max_c, 2) if f.temp_max_c == f.temp_max_c else None,
               "trace_source": tr.source, "backend": backend}
        epoch_rows.append(row)

    # write per-time trace
    trace_csv = f"data/train_traces/{out_prefix}.csv"
    epoch_idx = _epoch_index_for_samples(tr, bounds)
    pd.DataFrame({"t_s": tr.t, "power_w": tr.p_total / 1000.0,
                  "temp_c": (tr.temp if tr.temp is not None else np.nan),
                  "epoch": epoch_idx}).to_csv(trace_csv, index=False)

    # append per-epoch summary
    ep_csv = "data/train_epochs.csv"
    df = pd.DataFrame(epoch_rows)
    header = not os.path.exists(ep_csv)
    df.to_csv(ep_csv, mode="a", header=header, index=False)

    if save_fn:
        save_fn()
        print("[train] checkpoint saved.")
    print(f"[train] per-time trace -> {trace_csv}")
    print(f"[train] per-epoch summary appended -> {ep_csv}")
    print(f"[train] trace_source={tr.source} (ina3221=real, mock=synthetic)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18",
                    choices=["resnet18", "resnet50", "mobilenet_v3_large", "llama3.2-1b"])
    ap.add_argument("--precision", default="FP16")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--power-mode", default="MAXN")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 = full epoch")
    ap.add_argument("--imgsz", type=int, default=224, help="vision input size")
    ap.add_argument("--seq-len", type=int, default=128, help="llm block length")
    ap.add_argument("--period-ms", type=float, default=10.0)
    ap.add_argument("--save-ckpt", action="store_true")
    args = ap.parse_args()

    cfg = config.DeployConfig(model=args.model, precision=args.precision,
                              batch_size=args.batch_size, power_mode=args.power_mode,
                              mode="train", seq_len=args.seq_len)
    tag = cfg.tag()

    if not _TORCH:
        print("[train] torch not found -> MOCK training (synthetic power). Real numbers need the Orin.")
        train_mock(cfg, args.epochs, args.period_ms, tag)
        return
    if cfg.is_llm():
        train_llm(cfg, args.epochs, args.max_steps_per_epoch, args.save_ckpt, args.period_ms, tag)
    else:
        train_vision(cfg, args.epochs, args.max_steps_per_epoch, args.imgsz,
                     args.save_ckpt, args.period_ms, tag)


if __name__ == "__main__":
    main()
