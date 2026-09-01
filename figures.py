"""
figures.py — publication-grade, RIGOROUS figures from measured data only.
No synthetic fallbacks, no interpolated surfaces. Every figure uses real measured
rows/traces or is skipped with a loud note. Run: python3 figures.py
"""
import os, glob, sys, site
# Orin has two matplotlibs (system + pip --user); force the user copy's mpl_toolkits
# to the front so the 3D module doesn't load the stale system one and crash on import.
_us = site.getusersitepackages()
sys.path = [_us] + [p for p in sys.path if p != _us]
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
# NOTE: no explicit Axes3D import; add_subplot(projection="3d") registers it.

plt.rcParams.update({"font.size":11,"axes.grid":True,"grid.alpha":0.25,
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.titlesize":12,"axes.labelsize":11,"legend.fontsize":9})
FIG="results/figures"; os.makedirs(FIG,exist_ok=True)
PAL={"resnet18":"#3B6EA5","resnet50":"#8C4A9C","mobilenet_v3_large":"#C1662F",
     "efficientnet_b0":"#2E8B57","yolov8n":"#B03A2E"}
def col(m): return PAL.get(m,"#555555")
PB={"7W":7,"15W":15,"25W":25,"MAXN":40}

def load(path="data/waveforms.csv"):
    d=pd.read_csv(path)
    d=d[d["trace_source"]=="ina3221"]
    if not len(d): raise SystemExit(f"[figures] no real (ina3221) rows in {path}")
    d["pbudget"]=d.power_mode.map(PB); return d

def f_model_compare(d,metric,ylabel,fname,mode="inference"):
    pmodes=[p for p in ["7W","15W","25W","MAXN"] if p in d.power_mode.unique()]
    fig,axes=plt.subplots(1,len(pmodes),figsize=(4.3*len(pmodes),4.3),sharey=True,squeeze=False)
    for ax,pm in zip(axes[0],pmodes):
        s=d[(d.power_mode==pm)&(d["mode"]==mode)]
        for m in sorted(s.model.unique()):
            g=s[s.model==m].groupby("batch_size")[metric].agg(["mean","std"])
            ax.errorbar(g.index,g["mean"],yerr=g["std"].fillna(0),marker="o",capsize=4,lw=2,color=col(m),label=m)
        ax.set_title(pm); ax.set_xlabel("batch size"); ax.set_xscale("log",base=2)
        ax.set_xticks(sorted(s.batch_size.unique())); ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0][0].set_ylabel(ylabel); axes[0][0].legend()
    fig.suptitle(f"{ylabel} vs batch size by power mode ({mode}; mean±std over repeats)")
    fig.tight_layout(); fig.savefig(f"{FIG}/{fname}",dpi=150); plt.close(fig); print("  wrote",fname)

def f_precision(d,mode="inference"):
    precs=sorted(d.precision.unique())
    if len(precs)<2: print("  precision_compare: only",precs,"- skip (need FP32+FP16)"); return
    models=sorted(d.model.unique())
    fig,axes=plt.subplots(1,len(models),figsize=(4.3*len(models),4.3),sharey=True,squeeze=False)
    for ax,m in zip(axes[0],models):
        s=d[(d.model==m)&(d["mode"]==mode)&(d.power_mode=="MAXN")]
        for p in precs:
            g=s[s.precision==p].groupby("batch_size").p_peak_w.agg(["mean","std"])
            ax.errorbar(g.index,g["mean"],yerr=g["std"].fillna(0),marker="o",capsize=4,lw=2,label=p)
        ax.set_title(m);ax.set_xlabel("batch size")
    axes[0][0].set_ylabel("peak power (W)");axes[0][0].legend(title="precision")
    fig.suptitle("Precision effect on peak power (MAXN, inference)")
    fig.tight_layout();fig.savefig(f"{FIG}/precision_compare.png",dpi=150);plt.close(fig);print("  wrote precision_compare.png")

def f_scatter3d(d):
    fig=plt.figure(figsize=(9,6.5));ax=fig.add_subplot(111,projection="3d")
    mk={"inference":"o","train":"^"}
    for m in sorted(d.model.unique()):
        for mode in d["mode"].unique():
            s=d[(d.model==m)&(d["mode"]==mode)]
            if len(s): ax.scatter(s.p_peak_w,s.energy_per_inf_j,s.throughput_infps,c=col(m),
                marker=mk.get(mode,"o"),s=42,alpha=.85,edgecolor="k",linewidth=.3,label=f"{m}·{mode}")
    ax.set_xlabel("Peak power (W)");ax.set_ylabel("Energy/iter (J)");ax.set_zlabel("Throughput (it/s)")
    ax.set_title("Measured deployment points (real data only)");ax.legend(loc="upper left")
    fig.tight_layout();fig.savefig(f"{FIG}/scatter_3d.png",dpi=150);plt.close(fig);print("  wrote scatter_3d.png")

def f_grid_surface(d,mode="inference"):
    models=sorted(d[d["mode"]==mode].model.unique())
    fig=plt.figure(figsize=(5.5*max(len(models),1),5))
    drew=False
    for i,m in enumerate(models):
        s=d[(d.model==m)&(d["mode"]==mode)]
        piv=s.pivot_table(index="batch_size",columns="pbudget",values="p_peak_w",aggfunc="mean")
        if piv.shape[0]<2 or piv.shape[1]<2: continue
        X,Y=np.meshgrid(piv.columns.values.astype(float),piv.index.values.astype(float))
        ax=fig.add_subplot(1,len(models),i+1,projection="3d")
        ax.plot_surface(X,Y,piv.values,cmap="viridis",alpha=.8,edgecolor="k",linewidth=.4)
        ax.scatter(X,Y,piv.values,c="k",s=20)
        ax.set_xlabel("power budget (W)");ax.set_ylabel("batch");ax.set_zlabel("peak power (W)");ax.set_title(f"{m} ({mode})")
        drew=True
    if not drew: print("  grid_surface: grid too small - skip"); plt.close(fig); return
    fig.suptitle("Measured peak-power surface over (power × batch) — every vertex is real")
    fig.tight_layout();fig.savefig(f"{FIG}/grid_surface.png",dpi=150);plt.close(fig);print("  wrote grid_surface.png")

def f_per_epoch():
    p="data/train_epochs.csv"
    if not os.path.exists(p): print("  train_per_epoch: no train_epochs.csv - skip"); return
    df=pd.read_csv(p); df=df[df["trace_source"]=="ina3221"]
    if not len(df): print("  train_per_epoch: no real training epochs - skip"); return
    fig,ax1=plt.subplots(figsize=(8,4.6))
    for m in sorted(df.model.unique()):
        g=df[df.model==m].groupby("epoch").loss.mean()
        ax1.plot(g.index,g.values,"o-",color=col(m),label=f"{m} loss")
    ax1.set_xlabel("epoch");ax1.set_ylabel("training loss")
    ax2=ax1.twinx()
    for m in sorted(df.model.unique()):
        g=df[df.model==m].groupby("epoch").p_avg_w.mean()
        ax2.plot(g.index,g.values,"s--",color=col(m),alpha=.6)
    ax2.set_ylabel("avg power (W)  (dashed)")
    ax1.set_title("Per-epoch training: loss (solid) & avg power (dashed)");ax1.legend()
    fig.tight_layout();fig.savefig(f"{FIG}/train_per_epoch.png",dpi=150);plt.close(fig);print("  wrote train_per_epoch.png")

def f_power_over_time():
    tr=sorted(glob.glob("results/traces/*.csv"))
    if not tr: print("  power_over_time: no traces - skip"); return
    best,bestpk=None,-1
    for p in tr:
        if "_inference_" not in p: continue
        x=pd.read_csv(p)
        if len(x) and x.power_w.max()>bestpk: best,bestpk=p,x.power_w.max()
    if not best: best=tr[0]
    d=pd.read_csv(best); tag=os.path.basename(best).rsplit("_r",1)[0]
    fig,ax=plt.subplots(figsize=(8,4.2))
    ax.plot(d.t_s,d.power_w,lw=1.2,color="#3B6EA5"); ax.fill_between(d.t_s,d.power_w,alpha=.15,color="#3B6EA5")
    ax.set_xlabel("time (s)");ax.set_ylabel("power (W)")
    ax.set_title(f"Measured power waveform (highest-load inference)\n{tag}")
    fig.tight_layout();fig.savefig(f"{FIG}/power_over_time.png",dpi=150);plt.close(fig);print("  wrote power_over_time.png")

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/waveforms.csv",help="measured CSV to plot (real ina3221 rows)")
    ap.add_argument("--out-dir",default="results/figures",help="where to write the PNGs")
    args=ap.parse_args()
    global FIG; FIG=args.out_dir; os.makedirs(FIG,exist_ok=True)
    d=load(args.data); print(f"[figures] {len(d)} real rows | models={sorted(d.model.unique())} | precisions={sorted(d.precision.unique())}")
    for name, fn in [("compare_peak", lambda: f_model_compare(d,"p_peak_w","peak power (W)","compare_peak.png")),
                     ("compare_energy", lambda: f_model_compare(d,"energy_per_inf_j","energy/iter (J)","compare_energy.png")),
                     ("precision", lambda: f_precision(d)),
                     ("scatter_3d", lambda: f_scatter3d(d)),
                     ("grid_surface", lambda: f_grid_surface(d)),
                     ("per_epoch", f_per_epoch),
                     ("power_over_time", f_power_over_time)]:
        try: fn()
        except Exception as e: print(f"  [!] {name} failed: {e}")
    print("[figures] done ->",FIG)

if __name__=="__main__": main()
