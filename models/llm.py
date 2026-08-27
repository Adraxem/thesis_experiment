"""
models/llm.py — Llama-3.2-1B inference runner for the burst/decode regime (RQ1/RQ2).

The LLM matters here because its power waveform has TWO distinct phases the vision
models don't: a compute-heavy PREFILL spike (reads the whole prompt at once) and a
memory-bound DECODE ripple (one token at a time). We time them separately so the
sweep can expose that structure.

Backends, in order of preference:
  1. transformers + torch (real logits; CUDA on the Orin, CPU on the laptop)
  2. timing stub (no weights) — still exercises the prefill/decode loop shape

For maximum on-device fidelity, swap in TensorRT-LLM or llama.cpp (GGUF Q4_K_M)
at the marked TODO — the runner interface stays the same.
"""
from __future__ import annotations
import time

from .base import BaseRunner, RunResult, _have

_TF = _have("transformers") and _have("torch")

MODEL_ID = "meta-llama/Llama-3.2-1B"   # gated on HF; set HF_TOKEN or use a local path


class LLMRunner(BaseRunner):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model = None
        self.tok = None

    def load(self) -> None:
        if not _TF:
            self.backend = "stub"
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if (self.cfg.precision in ("FP16", "FP8", "INT8")
                                      and self.device == "cuda") else torch.float32
            self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, torch_dtype=dtype).to(self.device).eval()
            self.backend = f"transformers-{self.device}-{self.cfg.precision}"
            # TODO(Orin): for real INT4/INT8 decode power, load a TensorRT-LLM engine
            # or llama.cpp GGUF (Q4_K_M/Q8_0) instead of the HF fp16 model.
        except Exception as e:
            print(f"[llm] falling back to stub: {e}")
            self.backend = "stub"

    def run(self, n_iters: int) -> RunResult:
        """One 'inference' = one full prefill + gen_tokens decode steps."""
        sl, gt, bs = self.cfg.seq_len, self.cfg.gen_tokens, self.cfg.batch_size
        if self.model is None:
            # stub: model of prefill (O(seq)) + decode (O(tokens)) cost
            per = (0.0004 * sl + 0.0025 * gt) * bs
            prefill_s = decode_s = 0.0
            t0 = time.perf_counter()
            for _ in range(n_iters):
                time.sleep(0.0004 * sl * bs); prefill_s += 0.0004 * sl * bs
                time.sleep(0.0025 * gt * bs); decode_s += 0.0025 * gt * bs
            return RunResult(n_iters, time.perf_counter() - t0, "stub",
                             {"prefill_s": prefill_s, "decode_s": decode_s})

        import torch
        prompt = "The power waveform of an edge GPU " * max(1, sl // 8)
        enc = self.tok([prompt] * bs, return_tensors="pt", truncation=True,
                       max_length=sl).to(self.device)
        prefill_s = decode_s = 0.0
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                tp = time.perf_counter()
                out = self.model(**enc, use_cache=True)   # prefill
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                prefill_s += time.perf_counter() - tp
                td = time.perf_counter()
                for _ in range(gt):                        # decode
                    o = self.model(nxt, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nxt = o.logits[:, -1:].argmax(-1)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                decode_s += time.perf_counter() - td
        return RunResult(n_iters, time.perf_counter() - t0, self.backend,
                         {"prefill_s": prefill_s, "decode_s": decode_s})

    def train(self, n_iters: int) -> RunResult:
        """QLoRA-style fine-tune step loop on Llama-3.2-1B (single-device LLM training
        power). Uses PEFT LoRA if installed; else a full-param step; else a stub.
        Full pretraining of a large LLM does NOT fit a Nano — this is fine-tuning."""
        sl, bs = self.cfg.seq_len, self.cfg.batch_size
        if self.model is None:
            # training step ~ prefill fwd + bwd + opt; heavier than inference
            per = 0.0012 * sl * bs
            t0 = time.perf_counter()
            for _ in range(n_iters):
                time.sleep(per)
            return RunResult(n_iters, time.perf_counter() - t0, "stub-train",
                             {"note": "no backend/hardware; training timing stub"})
        import torch
        model = self.model
        try:
            from peft import LoraConfig, get_peft_model
            if not getattr(self, "_lora", False):
                lc = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
                model = get_peft_model(self.model, lc)
                self.model = model
                self._lora = True
            tag = self.backend + "-qlora"
        except Exception:
            tag = self.backend + "-train"   # full-param fallback
        model.train()
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-4)
        prompt = "The power waveform of an edge GPU " * max(1, sl // 8)
        enc = self.tok([prompt] * bs, return_tensors="pt", truncation=True,
                       max_length=sl).to(self.device)
        labels = enc["input_ids"].clone()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            opt.zero_grad()
            out = model(**enc, labels=labels)     # causal-LM loss
            out.loss.backward()
            opt.step()
            if self.device == "cuda":
                torch.cuda.synchronize()
        model.eval()
        return RunResult(n_iters, time.perf_counter() - t0, tag,
                         {"final_loss": float(out.loss.item())})
