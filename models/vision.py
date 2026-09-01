"""
models/vision.py — ResNet / MobileNet / YOLO runner (RQ1).

Two engines, chosen by the env var VISION_ENGINE (set by run_sweep's --engine flag):

  * "torch" (default): eager torch. Honest for FP32 / FP16 (.half()). It will REFUSE
    INT8 / INT4 / FP8 — running those on torch would execute FP32 kernels and mislabel
    the power as low-precision (a fake result). Refusing is the whole point.

  * "trt": build a REAL TensorRT engine at the requested precision via trtexec and run
    it (see models/trt_engine.py). This is the only honest INT8 / INT4 path. TensorRT is
    inference-only, so training always uses the torch path regardless of VISION_ENGINE.

Off-device (no torch) everything degrades to a timing stub so the pipeline still runs on
a laptop; those rows are tagged non-real by the telemetry layer (trace_source='mock').
"""
from __future__ import annotations
import os
import time

from .base import BaseRunner, RunResult, _have

_TORCH = _have("torch")
_TV = _have("torchvision")

# rough CPU per-iter seconds for the stub, scaled by model + batch
_STUB_COST = {"resnet18": 0.010, "resnet50": 0.028,
              "mobilenet_v3_large": 0.008, "yolov8n": 0.020}

_INT_PRECISIONS = ("INT8", "INT4", "FP8")


class VisionRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None
        self.input = None
        self.engine_path = None
        self._trt = False
        self.engine_mode = os.environ.get("VISION_ENGINE", "torch").lower()

    def _input_shape(self):
        if self.cfg.model.startswith("yolo"):
            return (self.cfg.batch_size, 3, 640, 640)
        return (self.cfg.batch_size, 3, 224, 224)

    # ---------------------------------------------------------------- load
    def load(self) -> None:
        # TensorRT path: inference only. Training falls through to the torch path below.
        if self.engine_mode == "trt" and self.cfg.mode == "inference":
            self._load_trt()
            return

        # Guard: INT8/INT4/FP8 are only real via TensorRT. Never fake them on torch
        # (this also skips INT8/INT4 *training*, which is not a real thing).
        if self.cfg.precision in _INT_PRECISIONS:
            raise RuntimeError(
                f"{self.cfg.precision} requires --engine trt (a real TensorRT engine). "
                f"Refusing to run it on the torch path, which would measure FP32 power and "
                f"mislabel it as {self.cfg.precision}.")

        if not _TORCH:
            self.backend = "stub"
            return
        import torch
        shape = self._input_shape()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.cfg.model.startswith("yolo"):
            if _have("ultralytics"):
                from ultralytics import YOLO
                self.model = YOLO(f"{self.cfg.model}.pt")
                self.backend = f"ultralytics-{self.device}"
            else:
                self.backend = "stub"
                return
        else:
            if not _TV:
                self.backend = "stub"
                return
            import torchvision.models as m
            builder = {"resnet18": m.resnet18, "resnet50": m.resnet50,
                       "mobilenet_v3_large": m.mobilenet_v3_large}[self.cfg.model]
            self.model = builder(weights=None).eval().to(self.device)
            self.input = torch.randn(*shape, device=self.device)
            if self.cfg.precision == "FP16" and self.device == "cuda":
                self.model = self.model.half()
                self.input = self.input.half()
            self.backend = f"torch-{self.device}-{self.cfg.precision}"

    # ---------------------------------------------------------------- TRT load
    def _load_trt(self) -> None:
        """Build (or reuse) a real TensorRT engine at cfg.precision and mark _trt."""
        from . import trt_engine
        if self.cfg.model.startswith("yolo"):
            # YOLO -> TRT is a separate export (ultralytics exports its own engine);
            # not wired here. Skip loudly so the sweep records it, never faked.
            raise RuntimeError("YOLO TensorRT export not implemented; run YOLO on --engine torch (FP16) only.")
        if not trt_engine.trt_available():
            raise RuntimeError("--engine trt requested but trtexec not found "
                               "(expected /usr/src/tensorrt/bin/trtexec on JetPack).")
        onnx_path = self._export_onnx_fp32()   # precision-agnostic FP32 ONNX (built once per model,batch)
        engine_path = trt_engine.engine_path_for(self.cfg.model, self.cfg.batch_size, self.cfg.precision)
        # build_engine raises TRTBuildError on unsupported precision -> sweep skips + reports.
        self.engine_path = trt_engine.build_engine(onnx_path, self.cfg.precision, engine_path)
        self.backend = f"trt-{self.cfg.precision.lower()}"
        self._trt = True

    def _export_onnx_fp32(self) -> str:
        """Export a clean FP32 ONNX (one per model+batch). TensorRT applies the precision
        at BUILD time, so the ONNX must NOT be pre-halved."""
        if not _TORCH:
            raise RuntimeError("torch required to export ONNX for the TensorRT build")
        import torch
        import torchvision.models as m
        os.makedirs("results/onnx", exist_ok=True)
        path = f"results/onnx/{self.cfg.model}_b{self.cfg.batch_size}_fp32.onnx"
        if os.path.exists(path):
            return path
        builder = {"resnet18": m.resnet18, "resnet50": m.resnet50,
                   "mobilenet_v3_large": m.mobilenet_v3_large}[self.cfg.model]
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net = builder(weights=None).eval().to(dev)
        dummy = torch.randn(*self._input_shape(), device=dev)
        torch.onnx.export(net, dummy, path, input_names=["input"],
                          output_names=["logits"], opset_version=17)
        print(f"[vision] exported FP32 ONNX -> {path}")
        return path

    # ---------------------------------------------------------------- run
    def run(self, n_iters: int) -> RunResult:
        if self._trt:
            from . import trt_engine
            warm = n_iters < 40   # measure_one calls run(warmup=10) first, then run(iters)
            res = trt_engine.run_engine(self.engine_path, warmup_ms=300,
                                        duration_s=(1.0 if warm else 6.0), iterations=n_iters)
            return RunResult(res.n_inf, res.wall_s, self.backend,
                             {"engine": self.engine_path, "qps": res.qps})

        if self.model is None:
            bs = self.cfg.batch_size
            return self._stub_run(n_iters, _STUB_COST.get(self.cfg.model, 0.015) * bs)
        import torch
        if self.cfg.model.startswith("yolo"):
            import numpy as np
            img = np.zeros((640, 640, 3), dtype="uint8")
            t0 = time.perf_counter()
            for _ in range(n_iters):
                self.model.predict(img, verbose=False, imgsz=640,
                                   batch=self.cfg.batch_size, device=self.device)
            return RunResult(n_iters, time.perf_counter() - t0, self.backend, {})
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                self.model(self.input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
        return RunResult(n_iters, time.perf_counter() - t0, self.backend,
                         {"engine": self.engine_path})

    # ---------------------------------------------------------------- train (torch only)
    def train(self, n_iters: int) -> RunResult:
        """Real forward+backward+optimizer loop (single-device training power).
        TensorRT is inference-only, so training is always torch; INT8/INT4 were already
        refused in load(). YOLO / no-torch degrade to a stub."""
        if self.model is None or self.cfg.model.startswith("yolo"):
            bs = self.cfg.batch_size
            return self._stub_run(n_iters, _STUB_COST.get(self.cfg.model, 0.015) * bs * 3.0)
        import torch
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        lossf = torch.nn.CrossEntropyLoss()
        x = self.input if self.input is not None else torch.randn(*self._input_shape(), device=self.device)
        if self.cfg.precision == "FP16" and self.device == "cuda":
            x = x.half()
        y = torch.randint(0, 1000, (self.cfg.batch_size,), device=self.device)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            opt.zero_grad()
            out = self.model(x)
            loss = lossf(out.float(), y)
            loss.backward()
            opt.step()
            if self.device == "cuda":
                torch.cuda.synchronize()
        self.model.eval()
        return RunResult(n_iters, time.perf_counter() - t0, self.backend + "-train",
                         {"final_loss": float(loss.item())})
