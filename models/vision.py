"""
models/vision.py — ResNet / MobileNet / YOLO inference runner.

Path on the Orin: torch (JetPack wheel) -> ONNX export -> TensorRT engine at the
requested precision (FP16/INT8/...). Off-device it runs eager torch on CPU, or a
timing stub if torch/torchvision are missing. The point is a real, repeatable
inference loop the PowerLogger can wrap.

TensorRT engine build is stubbed with a clear TODO where you plug `trtexec` or the
Python TensorRT API in once you're on the board — the ONNX export is real.
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


class VisionRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None
        self.input = None
        self.engine_path = None

    def _input_shape(self):
        if self.cfg.model.startswith("yolo"):
            return (self.cfg.batch_size, 3, 640, 640)
        return (self.cfg.batch_size, 3, 224, 224)

    def load(self) -> None:
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

        # On the Orin you would build a TensorRT engine here for the true peak-power
        # numbers. ONNX export path:
        self._maybe_export_onnx()

    def _maybe_export_onnx(self):
        if self.model is None or self.cfg.model.startswith("yolo") or not _TORCH:
            return
        import torch
        os.makedirs("results/onnx", exist_ok=True)
        path = f"results/onnx/{self.cfg.model}_b{self.cfg.batch_size}.onnx"
        self.engine_path = path
        if os.path.exists(path):
            return
        try:
            dummy = torch.randn(*self._input_shape(), device=self.device)
            if self.cfg.precision == "FP16" and self.device == "cuda":
                dummy = dummy.half()
            torch.onnx.export(self.model, dummy, path, input_names=["input"],
                              output_names=["logits"], opset_version=17)
            # TODO(Orin): build TensorRT engine, e.g.
            #   trtexec --onnx={path} --fp16|--int8 --saveEngine=...
            #   and run with the TensorRT Python runtime for real INT8/FP8 power.
        except Exception as e:
            print(f"[vision] onnx export skipped: {e}")

    def run(self, n_iters: int) -> RunResult:
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

    def train(self, n_iters: int) -> RunResult:
        """A real forward+backward+optimizer step loop (single-device training power).
        YOLO training is heavier; here we do a classification-style step for the CNNs
        and fall back to a stub for YOLO / when torch is unavailable."""
        if self.model is None or self.cfg.model.startswith("yolo"):
            bs = self.cfg.batch_size
            # training ~ 3x an inference step (fwd+bwd+opt)
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
