"""
run_sweep.py — RQ1 data-collection driver (resilient).
Per config: set power mode, load model, warm up, log INA3221 while running N
iters (inference or training), extract waveform features, append a row, and save
the raw per-time trace to results/traces/. A config that errors (e.g. CUDA OOM at
large batch) is recorded and SKIPPED — the sweep continues and reports failures at
the end, so one bad config never silently wrecks or aborts the whole run.
"""
from __future__ import annotations
import argparse, os, subprocess, time
import numpy as np, pandas as pd
import config
from models.base import make_runner
from power.telemetry import PowerLogger, IS_JETSON
from power.waveform_features import extract_features, WaveformFeatures

# Orin Nano (Super) nvpmodel ids (verify: grep POWER_MODEL /etc/nvpmodel.conf)
NVPMODEL_IDS = {"MAXN": 2, "25W": 1, "15W": 0}
TRACE_DIR = "results/traces"

def set_power_mode(mode):
    if not IS_JETSON: return
    pid = NVPMODEL_IDS.get(mode)
    if pid is None: print(f"[sweep] no nvpmodel id for '{mode}'"); return
    try:
        subprocess.run(["sudo","nvpmodel","-m",str(pid)],check=False); time.sleep(2)
    except Exception as e: print(f"[sweep] nvpmodel failed ({e})")

def _save_trace(trace,cfg,rep):
    os.makedirs(TRACE_DIR,exist_ok=True)
    pd.DataFrame({"t_s":trace.t,"power_w":trace.p_total/1000.0,
        "temp_c":(trace.temp if trace.temp is not None else np.full(len(trace.t),np.nan))}
        ).to_csv(f"{TRACE_DIR}/{cfg.tag()}_r{rep}.csv",index=False)

def measure_one(cfg,iters,warmup,period_ms,rep=0,save_traces=True):
    runner=make_runner(cfg); runner.load()
    if warmup: (runner.train if cfg.mode=="train" else runner.run)(warmup)
    with PowerLogger(period_ms=period_ms,cfg=cfg) as logger:
        res=runner.train(iters) if cfg.mode=="train" else runner.run(iters)
    trace=logger.trace(); feats=extract_features(trace,n_inferences=res.n_inferences)
    if save_traces: _save_trace(trace,cfg,rep)
    row=cfg.as_row(); row.update(feats.as_row())
    row["backend"]=res.backend; row["throughput_infps"]=res.n_inferences/max(res.wall_s,1e-9)
    row["trace_source"]=trace.source
    row.update({k:v for k,v in res.extra.items() if isinstance(v,(int,float))})
    return row

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="data/waveforms.csv")
    ap.add_argument("--iters",type=int,default=100); ap.add_argument("--warmup",type=int,default=10)
    ap.add_argument("--period-ms",type=float,default=5.0); ap.add_argument("--repeats",type=int,default=1)
    ap.add_argument("--models",nargs="*",default=None); ap.add_argument("--precisions",nargs="*",default=None)
    ap.add_argument("--batch-sizes",nargs="*",type=int,default=None); ap.add_argument("--power-modes",nargs="*",default=None)
    ap.add_argument("--modes",nargs="*",default=None)
    ap.add_argument("--save-traces",dest="save_traces",action="store_true",default=True)
    ap.add_argument("--no-save-traces",dest="save_traces",action="store_false")
    ap.add_argument("--smoke",action="store_true")
    args=ap.parse_args()

    spec=config.SweepSpec()
    if args.smoke:
        spec.models=["resnet18","llama3.2-1b"]; spec.precisions=["FP16","INT8"]
        spec.batch_sizes=[1,4]; spec.power_modes=["25W"]; spec.llm_gen_tokens=[16]
        args.iters,args.warmup=8,2
    if args.models: spec.models=args.models
    if args.precisions: spec.precisions=args.precisions
    if args.batch_sizes: spec.batch_sizes=args.batch_sizes
    if args.power_modes: spec.power_modes=args.power_modes
    if args.modes: spec.modes=args.modes

    cfgs=list(spec.iter_configs())
    print(f"[sweep] board={config.detect_board()} jetson={IS_JETSON} configs={len(cfgs)} x{args.repeats} iters={args.iters}")
    os.makedirs(os.path.dirname(args.out) or ".",exist_ok=True)
    rows,current_mode,failures=[],None,[]
    for i,cfg in enumerate(cfgs,1):
        if cfg.power_mode!=current_mode:
            set_power_mode(cfg.power_mode); current_mode=cfg.power_mode
        for rep in range(args.repeats):
            try:
                row=measure_one(cfg,args.iters,args.warmup,args.period_ms,rep=rep,save_traces=args.save_traces)
                row["repeat"]=rep; rows.append(row)
            except Exception as e:
                failures.append((cfg.tag(),rep,repr(e)))
                print(f"  [!] FAILED {cfg.tag()} rep{rep}: {e}")
                try:
                    import torch; torch.cuda.empty_cache()
                except Exception: pass
        if rows:
            print(f"  [{i}/{len(cfgs)}] {cfg.tag():46s} peak={rows[-1]['p_peak_w']:.1f}W "
                  f"E/it={rows[-1]['energy_per_inf_j']:.3f}J [{rows[-1]['backend']}]")
            pd.DataFrame(rows).to_csv(args.out,index=False)
    pd.DataFrame(rows).to_csv(args.out,index=False)
    print(f"[sweep] wrote {len(rows)} rows -> {args.out}  | traces -> {TRACE_DIR}/")
    if failures:
        print(f"[sweep] {len(failures)} config-runs FAILED and were skipped (recorded, not hidden):")
        for t,r,e in failures[:15]: print("   ",t,"rep",r,"->",e[:90])
    print(f"[sweep] targets: {WaveformFeatures.target_names()}")

if __name__=="__main__": main()
