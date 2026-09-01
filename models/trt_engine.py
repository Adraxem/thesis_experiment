"""
models/trt_engine.py — REAL TensorRT engine build + run via `trtexec` (RQ1, INT8/INT4).

Why this exists
---------------
The torch path can only do FP32 / FP16 honestly (`.half()`). INT8 / INT4 need an
actual quantized TensorRT engine. Running `--precisions INT8` on the torch path would
execute FP32 kernels and mislabel the power as INT8 — a fake result. This module builds
a genuine engine at the requested precision with `trtexec` (which ships with JetPack —
no pip, no apt) and runs it, so the measured waveform is the real low-precision one.

Design choices (all deliberate, all documented)
-----------------------------------------------
* `trtexec` is used for BOTH build and run. It is the standard NVIDIA tool, always
  present at /usr/src/tensorrt/bin/trtexec on JetPack, and it hammers the engine in a
  tight loop — exactly the steady-state workload whose power we want to sample.
* INT8 with NO calibration file: trtexec fabricates dynamic ranges. Accuracy is then
  garbage — but this thesis's *sweep* measures the MACHINE, not accuracy (see METHODS.md):
  the INT8 tensor-core kernels really run, so the INT8 *power waveform* is real. Accuracy
  is explicitly out of scope for the power sweep.
* INT4 is weight-only in TensorRT and normally needs explicit Q/DQ (NVIDIA ModelOpt).
  We ATTEMPT it via trtexec; if this TRT/board rejects it, the build fails LOUDLY and the
  sweep records the config as skipped — never a silent downgrade. (Ampere/Orin may not
  expose INT4 here; that skip is itself an honest finding.)

Nothing in this module runs off-Jetson: `trt_available()` returns False on a laptop and
callers fall back to the torch path.
"""
from __future__ import annotations
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass


class TRTUnavailable(RuntimeError):
    """trtexec not found on this machine (e.g. laptop / CI)."""


class TRTBuildError(RuntimeError):
    """trtexec failed to BUILD an engine at the requested precision (e.g. INT4 unsupported)."""


# Candidate locations for trtexec on JetPack.
_TRTEXEC_CANDIDATES = [
    "/usr/src/tensorrt/bin/trtexec",
    "/usr/local/tensorrt/bin/trtexec",
]

# Build flags per precision. FP32 = default (no flag). We keep --fp16 alongside --int8
# so layers TensorRT can't quantize fall back to fp16 (standard mixed-precision build).
_BUILD_FLAGS = {
    "FP32": [],
    "FP16": ["--fp16"],
    "INT8": ["--int8", "--fp16"],
    "FP8":  ["--fp8", "--fp16"],   # Hopper/Ada only; will fail on Ampere/Orin -> honest skip
    "INT4": ["--int4", "--fp16"],  # weight-only; may need ModelOpt Q/DQ -> honest skip if rejected
}


def trtexec_path() -> str | None:
    p = shutil.which("trtexec")
    if p:
        return p
    for c in _TRTEXEC_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def trt_available() -> bool:
    return trtexec_path() is not None


@dataclass
class TRTRunResult:
    qps: float          # steady-state throughput (inferences/sec) reported by trtexec
    wall_s: float       # wall time of the timed run (Python-measured, wraps the trace)
    n_inf: int          # inferences in the timed window (qps * wall_s)


def engine_path_for(model: str, batch: int, precision: str, out_dir: str = "results/engines") -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{model}_b{batch}_{precision}.engine")


def build_engine(onnx_path: str, precision: str, engine_path: str,
                 workspace_mb: int = 2048, timeout_s: int = 1800) -> str:
    """Build a TensorRT engine from an FP32 ONNX at `precision`. Cached: if the engine
    file already exists it is reused (engines are power-mode- and repeat-independent).
    Raises TRTBuildError on failure — the caller (sweep) then SKIPS and reports the config."""
    exe = trtexec_path()
    if exe is None:
        raise TRTUnavailable("trtexec not found (not on a JetPack device?)")
    if precision not in _BUILD_FLAGS:
        raise TRTBuildError(f"unknown precision {precision!r}")
    if os.path.exists(engine_path) and os.path.getsize(engine_path) > 0:
        return engine_path  # cached

    cmd = [exe, f"--onnx={onnx_path}", f"--saveEngine={engine_path}",
           f"--memPoolSize=workspace:{workspace_mb}", "--skipInference"] + _BUILD_FLAGS[precision]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    dt = time.perf_counter() - t0
    if proc.returncode != 0 or not (os.path.exists(engine_path) and os.path.getsize(engine_path) > 0):
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise TRTBuildError(
            f"trtexec build FAILED for {precision} ({onnx_path}) rc={proc.returncode} "
            f"in {dt:.0f}s. This precision may be unsupported on this board. "
            f"trtexec tail: {' | '.join(tail)}")
    print(f"[trt] built {precision} engine in {dt:.0f}s -> {engine_path}")
    return engine_path


_QPS_RE = re.compile(r"Throughput:\s*([0-9.]+)\s*qps", re.IGNORECASE)


def run_engine(engine_path: str, warmup_ms: int = 300, duration_s: float = 6.0,
               iterations: int = 150, timeout_s: int = 600) -> TRTRunResult:
    """Run a built engine in a tight loop for ~duration_s while the caller's PowerLogger
    samples the INA3221. Returns real steady-state throughput parsed from trtexec."""
    exe = trtexec_path()
    if exe is None:
        raise TRTUnavailable("trtexec not found")
    cmd = [exe, f"--loadEngine={engine_path}", f"--warmUp={warmup_ms}",
           f"--duration={duration_s}", f"--iterations={max(iterations, 1)}",
           "--useSpinWait", "--noDataTransfers"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise TRTBuildError(f"trtexec run failed rc={proc.returncode}: {' | '.join(tail)}")
    m = _QPS_RE.search(proc.stdout or "")
    qps = float(m.group(1)) if m else float(iterations) / max(wall, 1e-9)
    n_inf = int(round(qps * wall))
    return TRTRunResult(qps=qps, wall_s=wall, n_inf=max(n_inf, 1))
