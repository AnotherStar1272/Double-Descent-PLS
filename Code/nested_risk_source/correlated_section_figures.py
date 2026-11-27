"""AR(.7), G=H: high-depth proportional limits and paired simulations.

No simulated spectra enter the theory. Lagrange inversion evaluates the
rooted moment series in polynomial time, avoiding Catalan enumeration.
For G=H, cyclicity gives E[a,b]=D[a+b], C=D and N=M.
Ridge uses the oracle limiting penalty gamma/SNR, not sample-risk tuning.
"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
import mpmath as mp
import numpy as np
import pandas as pd
from scipy import linalg, optimize
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from models import pls_krylov_path
from structured_covariance import ar_moment, population, ols_limit, ols_multiple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "Experiment/outputs_correlated_section"
OLD = ROOT / "Experiment/outputs_structured_covariance"
FIG = ROOT / "Experiment/theory"
COUNTS = (1, 2, 5, 10, 20)
SNRS = (1., 5.)
_ratios = np.geomspace(.2,3,36)
for _g in (.5,1.,2.):
    _ratios[np.argmin(abs(np.log(_ratios/_g)))]=_g
RATIOS = tuple(np.round(400*_ratios).astype(int)/400)
COLORS = {1:"black", 2:"red", 5:"#00B800", 10:"blue", 20:"#984ea3"}


def moments(gamma, maximum=41, rho=.7):
    """Lagrange coefficients: y=z*(1+gamma*sum(mu[l]*y**l)).

    M[q]=[z**q] sum(mu[l]*y**l);
    D[q]=[z**q] sum(mu[l+1]*y**l).
    """
    g = mp.mpf(str(gamma))
    mu = [ar_moment(j, rho) for j in range(maximum+2)]
    phi = [mp.mpf(1)] + [g*x for x in mu[1:maximum+1]]
    power = [mp.mpf(1)] + [mp.mpf(0)]*(maximum-1)
    M, D = [mp.mpf(1)], [mu[1]]
    for q in range(1, maximum+1):
        power = [sum(power[j]*phi[i-j] for j in range(i+1))
                 for i in range(maximum)]
        M.append(sum(j*mu[j]*power[q-j] for j in range(1,q+1))/q)
        D.append(sum(j*mu[j+1]*power[q-j] for j in range(1,q+1))/q)
    return M, D


def pls_theory(gamma, snrs=SNRS, counts=COUNTS, dps=100, rho=.7):
    with mp.workdps(dps):
        M,D = moments(gamma, 2*max(counts)+1, rho)
        g = mp.mpf(str(gamma))
        rows = []
        for snr in snrs:
            r = mp.mpf(str(snr))  # mu1=1, sigma2=1, G=H
            for k in counts:
                A = mp.matrix(k); B = mp.matrix(k)
                a = mp.matrix(k,1); b = mp.matrix(k,1)
                for i in range(k):
                    a[i] = r*M[i+2]+g*M[i+1]
                    b[i] = r*D[i+1]
                    for j in range(k):
                        A[i,j] = r*M[i+j+3]+g*M[i+j+2]
                        B[i,j] = r*D[i+j+2]+g*D[i+j+1]
                v = mp.lu_solve(A,a)
                risk = r-2*(v.T*b)[0]+(v.T*B*v)[0]
                assert risk > 0
                rows.append(dict(gamma=gamma,snr=snr,method=f"PLS{k}",
                                 theory=float(risk)))
        return rows


def ridge_limit(gamma, snr, penalty=None, rho=.7):
    """Trace resolvent for G=H; includes bias and variance for any lambda>0.

    J(u)=int t/(t+u)dH, lambda=u*(1-gamma*J), e=J/(1-gamma*J).
    R=-r2*lambda^2*e' + gamma*(e+lambda*e').
    At lambda=gamma/r2 this simplifies to gamma*e.
    AR law gives J=1/sqrt(1+2*x*u+u^2), x=(1+rho^2)/(1-rho^2).
    """
    lam = gamma/snr if penalty is None else penalty
    x = (1+rho*rho)/(1-rho*rho)
    def J(u): return 1/np.sqrt(1+2*x*u+u*u)
    u = np.exp(optimize.brentq(lambda z: np.exp(z)*(1-gamma*J(np.exp(z)))-lam,
                              -35,35,xtol=1e-13))
    j = J(u); a = 1-gamma*j
    jp = -(x+u)*j**3
    # Algebraically cancel the divergent terms before evaluating near lambda=0.
    return float((-snr*u*u*jp+gamma*(j+u*jp))/(a-gamma*u*jp))


def validate():
    from correlated_theory_extension import fast_terms, _evaluate
    from structured_covariance import population_moments
    errors = []
    with mp.workdps(90):
        mu,nu = population_moments("ar07","balanced")
        for g in (.5,1.,2.):
            M,D = moments(g,9)
            for q in range(1,8):
                errors += [float(abs(M[q]/_evaluate(fast_terms(q),mp.mpf(str(g)),mu,nu)-1)),
                           float(abs(D[q]/_evaluate(fast_terms(q,0),mp.mpf(str(g)),mu,nu)-1))]
            for a,b in ((1,1),(2,3),(3,3)):
                errors.append(float(abs(D[a+b]/_evaluate(fast_terms(a+b,a),mp.mpf(str(g)),mu,nu)-1)))
    assert max(errors)<1e-70
    precision=[]; isotropic=[]; ridge=[]
    from plot_pls_snr_comparison import high_precision_risk
    for g in (.2,.99,1.,1.01,3.):
        a=pls_theory(g,dps=100); b=pls_theory(g,dps=160)
        precision += [abs(x["theory"]/y["theory"]-1) for x,y in zip(a,b)]
        for row in pls_theory(g,counts=(1,5,20),rho=0.):
            ref=high_precision_risk(g,row["snr"],int(row["method"][3:]))
            isotropic.append(abs(row["theory"]/ref-1))
        for s in SNRS:
            optimum=optimize.minimize_scalar(lambda z:ridge_limit(g,s,np.exp(z)),bounds=(-12,8),method="bounded")
            ridge.append(abs(np.exp(optimum.x)/(g/s)-1))
            if g!=1:
                assert abs(ridge_limit(g,s,1e-7)/ols_limit(g,"ar07","balanced",s)-1)<.01
    assert max(precision)<1e-10 and max(isotropic)<1e-9 and max(ridge)<1e-4
    result=dict(rooted_moment_relative_error=max(errors),precision_100_vs_160=max(precision),
                isotropic_relative_error=max(isotropic),ridge_optimum_relative_error=max(ridge))
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(result,flush=True)


def theory_grid():
    grid=np.unique(np.r_[np.geomspace(.2,3,111),np.linspace(.8,1.2,33),RATIOS])
    records=[]; start=time.monotonic()
    for i,g in enumerate(grid):
        cache=OUT/"theory_checkpoints"/f"{g:.14g}.json"
        if cache.exists(): rows=json.loads(cache.read_text())
        else:
            rows=pls_theory(float(g))
            for s in SNRS:
                rows.append(dict(gamma=float(g),snr=s,method="Ridge",theory=ridge_limit(g,s)))
                if g!=1:
                    rows.append(dict(gamma=float(g),snr=s,method="OLS",theory=ols_limit(g,"ar07","balanced",s)))
            cache.parent.mkdir(parents=True,exist_ok=True)
            cache.write_text(json.dumps(rows))
        records.extend(rows)
        if i%15==0: print(f"theory {i+1}/{len(grid)} {time.monotonic()-start:.1f}s",flush=True)
    pd.DataFrame(records).to_csv(OUT/"theory.csv",index=False)


def sample(n,p,rep):
    # Reproduce the previous suite's exact seed and balanced response pairing.
    words=np.frombuffer(sha256(f"2026091707/ar07/{n}/{p}/{rep}".encode()).digest(),dtype="<u4")
    rng=np.random.default_rng(np.random.SeedSequence(words))
    eig,beta0=population("ar07",p,"balanced")
    X=rng.standard_normal((n,p))*np.sqrt(eig)
    noise=rng.standard_normal(n)
    Y=np.column_stack([X@(beta0*np.sqrt(s))+noise for s in SNRS])
    ols=ols_multiple(X,Y)
    rows=[]
    for col,snr in enumerate(SNRS):
        beta=beta0*np.sqrt(snr); y=Y[:,col]
        path=pls_krylov_path(X,y,COUNTS)
        assert all(path.effective_k[k]==k for k in COUNTS)
        lam=(p/n)/snr
        if n>=p:
            A=X.T@X/n; A.flat[::p+1]+=lam
            br=linalg.solve(A,X.T@y/n,assume_a="pos",check_finite=False)
        else:
            A=X@X.T/n; A.flat[::n+1]+=lam
            br=X.T@linalg.solve(A,y,assume_a="pos",check_finite=False)/n
        for method,bhat in [(f"PLS{k}",path.estimates[k]) for k in COUNTS]+[("Ridge",br),("OLS",ols[:,col])]:
            delta=bhat-beta
            rows.append(dict(n=n,p=p,gamma=p/n,snr=snr,rep=rep,method=method,
                             generalization_error=float(eig@(delta*delta))))
    return rows


def simulate(workers=4):
    cells=sorted({(400,round(400*g)) for g in RATIOS}|
                 {(n,round(n*g)) for n in (200,400,800) for g in (.5,1.,2.)})
    start=time.monotonic()
    for n,p in cells:
        target=OUT/"simulation_checkpoints"/f"n{n}_p{p}.csv"
        if target.exists() and "OLS" in set(pd.read_csv(target).method): continue
        rows=[]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(sample,n,p,r) for r in range(100)]
            for f in as_completed(futures): rows.extend(f.result())
        target.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).sort_values(["rep","snr","method"]).to_csv(target,index=False)
        print(f"simulation n={n} p={p} elapsed={time.monotonic()-start:.1f}s",flush=True)
    raw=pd.concat([pd.read_csv(OUT/"simulation_checkpoints"/f"n{n}_p{p}.csv") for n,p in cells])
    old=pd.read_csv(OLD/"raw.csv")
    old=old[(old.covariance=="ar07")&(old.alignment=="balanced")]
    paired=raw.merge(old,on=["n","p","gamma","snr","rep","method"],suffixes=("","_old"))
    paired=paired[paired.method.str.startswith("PLS")]
    replay=float(np.max(abs(paired.generalization_error-paired.generalization_error_old)))
    assert len(paired)>0 and replay<1e-9
    raw.to_csv(OUT/"raw.csv",index=False)
    summary=raw.groupby(["n","p","gamma","snr","method"]).generalization_error.agg(["mean","std","count"]).reset_index()
    summary["se"]=summary["std"]/np.sqrt(summary["count"])
    assert summary["count"].eq(100).all()
    summary.to_csv(OUT/"summary.csv",index=False)
    (OUT/"simulation_validation.json").write_text(json.dumps(dict(paired_replay_max_abs_error=replay,
        cells=len(cells),repetitions=100,n_baseline=400,seed=2026091707,counts=COUNTS),indent=2))


def style(ax):
    ax.set(xlabel=r"$\gamma$",ylabel="",xscale="log",xlim=(.19,3.15))
    ax.xaxis.set_major_locator(FixedLocator([.2,.5,1,2,3]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(1,color=".7",ls=":",lw=.8)


def plots():
    theory=pd.read_csv(OUT/"theory.csv"); sim=pd.read_csv(OUT/"summary.csv")
    plt.rcParams.update({"font.family":"serif","mathtext.fontset":"cm","font.size":11,
                         "axes.labelsize":13,"legend.fontsize":9})
    FIG.mkdir(parents=True,exist_ok=True)
    saved=[]
    fig,axes=plt.subplots(2,3,figsize=(10,6.0),layout="constrained")
    gray_styles = {1:("0.05","o","-"), 2:("0.25","s","--"),
                   5:("0.45","^","-."), 10:("0.6","D",":"),
                   20:("0.1","v",(0,(5,2,1,2,1,2)))}
    for row,s in enumerate(SNRS):
        for col,g in enumerate((.5,1.,2.)):
            ax=axes[row,col]
            for k in COUNTS:
                a=sim[(sim.snr==s)&(sim.gamma==g)&(sim.method==f"PLS{k}")].sort_values("n")
                value=theory[(theory.snr==s)&np.isclose(theory.gamma,g,rtol=0,atol=1e-12)&(theory.method==f"PLS{k}")].theory.iloc[0]
                gray,marker,ls = gray_styles[k]
                ax.axhline(value,color=gray,ls=ls,lw=1.1)
                ax.errorbar(a.n,a["mean"],yerr=2*a.se,fmt=marker,color=gray,
                            mfc="white",ms=4,capsize=2,label=rf"$k={k}$")
            ax.set(xlabel=r"$n$",ylabel="",xticks=[200,400,800],xlim=(140,860))
            ax.text(.98,.96,f"({chr(97+row*3+col)})",transform=ax.transAxes,va="top",ha="right",
                    bbox=dict(facecolor="white",edgecolor="none",pad=1))
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=gray_styles[k][0],marker=gray_styles[k][1],
                    ls=gray_styles[k][2],mfc="white",label=rf"$k={k}$") for k in COUNTS]
    fig.legend(handles=handles,loc="upper center",ncol=5,frameon=False)
    fig.get_layout_engine().set(rect=(0,0,1,.94))
    path=FIG/"ar07_convergence_extended.png"; fig.savefig(path,dpi=300,facecolor="white")
    plt.close(fig); saved.append(path)
    for path in saved: shutil.copy2(path,ROOT/"Paper/Figs"/path.name)
    joined=sim.merge(theory,on=["gamma","snr","method"],how="left")
    joined["relative_discrepancy"]=abs(joined["mean"]/joined.theory-1)
    joined.to_csv(OUT/"agreement.csv",index=False)
    report=dict(counts=COUNTS,snrs=SNRS,rho=.7,dps=100,
                ridge_penalty="gamma/SNR; oracle limit, not cross-validation",baseline_n=400,
                replicates=100,baseline_ratios=list(RATIOS),
                source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                files={p.name:sha256(p.read_bytes()).hexdigest() for p in saved})
    (OUT/"manifest.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["validate","theory","simulate","plot"])
    parser.add_argument("--workers",type=int,default=4); args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with threadpool_limits(limits=1):
        {"validate":validate,"theory":theory_grid,"simulate":lambda:simulate(args.workers),"plot":plots}[args.mode]()
