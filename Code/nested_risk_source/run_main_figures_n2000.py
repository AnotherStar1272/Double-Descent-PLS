"""Reproduce every main-text comparison at n=2000, retaining historical data.

Dense Gaussian samples are generated, not limiting/sample surrogate spectra.
One sample eigendecomposition is reused across isotropic SNRs and omitted
signal settings. Orthogonal spectral Krylov fits are exactly the same PLS
least-squares problem. AR samples use the existing independently audited code.
Checkpoints contain 10 independent paired repetitions and permit resumption.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import correlated_section_figures as ar
from run_misspecified_a3_k20 import spectral_estimates
from models import pls_krylov_path, minnorm_ols

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Experiment/outputs_main_n2000"
N = 2000
COUNTS = (1, 2, 5, 10, 20)
SNRS = (1., 2.33, 3.66, 5.)
COLORS = {1:"black", 2:"red", 5:"#00B800", 10:"blue", 20:"#984ea3"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def spectral_samples(X, omitted, noise, rep):
    n, p = X.shape
    # Coordinates of e_1 and the independent Gaussian errors in the sample SVD.
    if p <= n:
        lam, U = eigh(X.T @ X / n, check_finite=False)
        coefficient = U[0].copy()
        eps = U.T @ (X.T @ noise / n)
        extra = U.T @ (X.T @ omitted / n)
        null_unit = 0.
    else:
        lam, U = eigh(X @ X.T / n, check_finite=False)
        coefficient = (U.T @ X[:, 0]) / np.sqrt(n * lam)
        eps = np.sqrt(lam / n) * (U.T @ noise)
        extra = np.sqrt(lam / n) * (U.T @ omitted)
        null_unit = max(0., 1. - coefficient @ coefficient)
    assert lam.min() > 0
    configurations = [("iso", s, s, 0., (1, 2, 5, 10)) for s in SNRS]
    configurations += [(f"a{a}", 5., 5.-5.*(1+p/n)**(-a),
                        5.*(1+p/n)**(-a), COUNTS) for a in (3, 5)]
    rows = []
    for model, snr, signal, missing, counts in configurations:
        beta = np.sqrt(signal) * coefficient
        rhs = lam * beta + eps + np.sqrt(missing) * extra
        fits = spectral_estimates(lam, rhs)
        fits[0] = rhs / lam
        for k in (*counts, 0):
            risk = float(np.sum((fits[k]-beta)**2) + signal*null_unit + missing)
            rows.append(dict(n=n,p=p,gamma=p/n,model=model,snr=snr,k=k,
                             method=f"PLS{k}" if k else "OLS",rep=rep,risk=risk))
    return rows


def iso_sample(n, p, rep):
    rng = np.random.default_rng(np.random.SeedSequence([2026093001,n,p,rep]))
    X = rng.standard_normal((n,p))
    omitted, noise = rng.standard_normal((2,n))
    return spectral_samples(X, omitted, noise, rep)


def ar_sample(n,p,rep):
    rows = ar.sample(n,p,rep)
    for row in rows:
        row["model"] = "ar07"
        row["risk"] = row.pop("generalization_error")
        row["k"] = int(row["method"][3:]) if row["method"].startswith("PLS") else 0
    return rows


def validate():
    maximum = 0.
    rng = np.random.default_rng(302026)
    for n,p in ((48,32),(48,70)):
        X=rng.standard_normal((n,p)); omitted,noise=rng.standard_normal((2,n))
        data=pd.DataFrame(spectral_samples(X,omitted,noise,0))
        for model in ("iso","a3","a5"):
            settings=SNRS if model=="iso" else (5.,)
            for s in settings:
                missing=0. if model=="iso" else 5.*(1+p/n)**(-int(model[1:]))
                beta=np.zeros(p); beta[0]=np.sqrt(s-missing)
                y=X@beta+noise+np.sqrt(missing)*omitted
                counts=(1,2,5,10) if model=="iso" else COUNTS
                fits=pls_krylov_path(X,y,counts).estimates
                fits[0]=minnorm_ols(X,y)
                for k, bhat in fits.items():
                    reference=float(np.sum((bhat-beta)**2)+missing)
                    actual=data[(data.model==model)&(data.snr==s)&(data.k==k)].risk.iloc[0]
                    maximum=max(maximum,abs(actual-reference)/max(1.,reference))
    assert maximum<1e-9, maximum
    result={"max_relative_difference_from_direct_PLS_and_OLS":maximum,"passed":True}
    save_json(OUT/"estimator_validation.json",result)
    print(result,flush=True)


def jobs():
    tasks=[]
    for p in np.rint(N*np.geomspace(.1,10,36)).astype(int):
        for start in range(0,200,10): tasks.append(("iso",int(p),start))
    for g in ar.RATIOS:
        for start in range(0,100,10): tasks.append(("ar07",int(round(N*g)),start))
    return tasks


def checkpoint(task):
    model,p,start=task
    return OUT/"checkpoints"/f"{model}_n{N}_p{p}_r{start:03d}.csv"


def batch(task):
    model,p,start=task
    rows=[]
    function=iso_sample if model=="iso" else ar_sample
    for rep in range(start,start+10): rows.extend(function(N,p,rep))
    frame=pd.DataFrame(rows)
    assert np.isfinite(frame.risk).all() and frame.risk.ge(0).all()
    assert len(frame)==(320 if model=="iso" else 140)
    path=checkpoint(task)
    temporary=path.with_suffix(".partial")
    frame.to_csv(temporary,index=False); temporary.replace(path)
    return task


def simulate(workers):
    (OUT/"checkpoints").mkdir(parents=True,exist_ok=True)
    validate()
    tasks=jobs(); pending=[]
    for task in tasks:
        path=checkpoint(task)
        if path.exists():
            saved=pd.read_csv(path)
            assert len(saved)==(320 if task[0]=="iso" else 140)
            assert saved.n.eq(N).all() and saved.rep.nunique()==10
        else: pending.append(task)
    print(f"Resume: {len(tasks)-len(pending)}/{len(tasks)} completed batches",flush=True)
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(batch,t) for t in pending]
        for number,future in enumerate(as_completed(futures),1):
            task=future.result()
            elapsed=time.monotonic()-started
            print(f"{number}/{len(pending)} new batches; {task}; elapsed {elapsed/60:.1f} min",flush=True)
            save_json(OUT/"progress.json",dict(completed=len(tasks)-len(pending)+number,
                       total=len(tasks),elapsed_seconds=elapsed,last_task=task))
    aggregate()


def aggregate():
    raw=pd.concat([pd.read_csv(checkpoint(t)) for t in jobs()],ignore_index=True)
    assert len(raw)==280800 and not raw.duplicated(["model","p","snr","method","rep"]).any()
    assert raw.n.eq(N).all() and np.isfinite(raw.risk).all() and raw.risk.ge(0).all()
    summary=raw.groupby(["n","p","gamma","model","snr","k","method"],as_index=False).agg(
        mean=("risk","mean"),std=("risk","std"),repeats=("risk","size"))
    assert summary.loc[summary.model=="ar07","repeats"].eq(100).all()
    assert summary.loc[summary.model!="ar07","repeats"].eq(200).all()
    raw.to_csv(OUT/"raw.csv",index=False); summary.to_csv(OUT/"summary.csv",index=False)
    save_json(OUT/"simulation_manifest.json",dict(passed=True,n=N,raw_rows=len(raw),
        independent_samples_isotropic=7200,independent_samples_AR=3600,
        repetitions_isotropic_and_misspecified=200,repetitions_AR=100,
        sampling="Dense Gaussian X; exact sample eigendecomposition for isotropic fits; existing AR code",
        pairing="All estimators and SNRs paired; a=3,5 share isotropic X, omitted covariate and noise",
        files_sha256={name:digest(OUT/name) for name in ("raw.csv","summary.csv")},
        source_sha256={p.name:digest(p) for p in (Path(__file__),Path(ar.__file__),
                     Path(__file__).with_name("models.py"),Path(__file__).with_name("run_misspecified_a3_k20.py"),
                     Path(__file__).with_name("structured_covariance.py"))}))


def plots():
    report=json.loads((OUT/"simulation_manifest.json").read_text())
    assert report["passed"] and report["n"]==N
    for name,value in report["files_sha256"].items(): assert digest(OUT/name)==value
    for name,value in report["source_sha256"].items(): assert digest(Path(__file__).with_name(name))==value
    summary=pd.read_csv(OUT/"summary.csv")
    figdir=OUT/"figures"; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":11,
                        "axes.labelsize":13,"legend.fontsize":10})
    configurations=[("iso",s,f"pls_ols_snr_{str(s).replace('.','_').removesuffix('_0')}") for s in SNRS]
    configurations += [(f"a{a}",5.,f"misspecified_pls_ols_a_{a}_k20") for a in (3,5)]
    configurations += [("ar07",s,f"ar07_pls_ols_ridge_snr_{int(s)}") for s in (1.,5.)]
    artifacts=[]; inputs={}
    for model,snr,basename in configurations:
        if model=="iso":
            source=ROOT/"Experiment/outputs_snr_comparison/pls_ols_snr_theory.csv"
            t=pd.read_csv(source).query("snr==@snr").rename(columns={"risk":"theory"})
        elif model.startswith("a") and model!="ar07":
            source=ROOT/f"Experiment/outputs_misspecified_{model}_k20/theory.csv"
            t=pd.read_csv(source).rename(columns={"error":"theory"})
        else:
            source=ROOT/"Experiment/outputs_correlated_section/theory.csv"
            t=pd.read_csv(source).query("snr==@snr")
        inputs[source.relative_to(ROOT).as_posix()]=digest(source)
        if "method" not in t: t["method"]=t.k.map(lambda k:f"PLS{k}" if k else "OLS")
        points=summary[(summary.model==model)&(summary.snr==snr)]
        fig,ax=plt.subplots(figsize=(7.2,5.6),layout="constrained")
        counts=(1,2,5,10) if model=="iso" else COUNTS
        legend_markers={} if model=="ar07" else dict(marker="o",markevery=[],ms=3.5)
        for k in counts:
            curve=t[t.method==f"PLS{k}"].sort_values("gamma")
            sample=points[points.method==f"PLS{k}"].sort_values("gamma")
            color="#A020B0" if k==20 and model!="ar07" else COLORS[k]
            ax.plot(curve.gamma,curve.theory,color=color,lw=1.65,label=rf"PLS $k={k}$",**legend_markers)
            ax.plot(sample.gamma,sample["mean"],"o",color=color,ms=3.5)
        for method,color,ls,marker in (("OLS",".35","--","o"),("Ridge","#e68a00","-.","s")):
            curve=t[t.method==method].sort_values("gamma")
            if curve.empty: continue
            if method=="OLS":
                for mask,label in ((curve.gamma<1,"OLS"),(curve.gamma>1,None)):
                    ax.plot(curve.loc[mask,"gamma"],curve.loc[mask,"theory"],color=color,ls=ls,lw=1.6,
                            label=label,markerfacecolor="white",**legend_markers)
            else: ax.plot(curve.gamma,curve.theory,color=color,ls=ls,lw=1.7,label="Optimal ridge")
            sample=points[(points.method==method)&((points.gamma!=1)|(method!="OLS"))].sort_values("gamma")
            ax.plot(sample.gamma,sample["mean"],ls="none",marker=marker,mfc="white",color=color,ms=3.5)
        ax.axhline(snr,color=".5",ls=":",lw=1.2,label="Null")
        ax.axvline(1,color=".7",ls=":",lw=.8)
        upper=(7 if snr==5 else 5) if model=="iso" else 12
        if model=="ar07":
            upper=max(5 if snr==1 else 7,np.ceil(1.08*max(t[t.method.str.startswith("PLS")].theory.max(),points[points.method.str.startswith("PLS")]["mean"].max())))
        assert t[t.method.str.startswith("PLS")].theory.max() < upper
        assert points[points.method.str.startswith("PLS")]["mean"].max() < upper
        ax.set(xscale="log",xlim=(.19,3.15) if model=="ar07" else (.085,12),
               ylim=(0,upper),xlabel=r"$\gamma$",ylabel="")
        ticks=[.2,.5,1,2,3] if model=="ar07" else [.1,.2,.5,1,2,5,10]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.legend(loc="upper left",ncol=2 if model=="ar07" else 1,framealpha=1,
                  fancybox=False,edgecolor="black",fontsize=9 if model=="ar07" else 10)
        path=figdir/f"{basename}_n2000.png"
        fig.savefig(path,dpi=300,facecolor="white"); plt.close(fig)
        shutil.copy2(path,ROOT/"Paper/Figs"/path.name)
        artifacts.append(dict(file=path.name,sha256=digest(path)))
    assert len(artifacts)==8
    save_json(OUT/"figure_manifest.json",dict(n=N,figures=artifacts,theory_inputs=inputs,
              simulation_manifest_sha256=digest(OUT/"simulation_manifest.json")))
    print("Published eight n=2000 figures; historical figures retained.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("validate","simulate","aggregate","plot","benchmark"))
    parser.add_argument("--workers",type=int,default=3)
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        if args.mode=="benchmark":
            for model,p in (("iso",2000),("iso",20000),("ar07",2000),("ar07",6000)):
                start=time.monotonic()
                (iso_sample if model=="iso" else ar_sample)(N,p,9999)
                print(model,p,"seconds",time.monotonic()-start,flush=True)
        else: {"validate":validate,"simulate":lambda:simulate(args.workers),
               "aggregate":aggregate,"plot":plots}[args.mode]()
