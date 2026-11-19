"""
tri_all and the cosine signals under FLTrust step parity, root 100, seeds 72-76.
Plan and reading rule: results/tri_all_step_parity/PLAN.md. Writes results/tri_all_step_parity/analysis/.

Parity runs: results/tri_all_step_parity/raw/<ds>/<system>@stmedian/. Native references: the B1 runs,
results/inclusion_mechanism/raw/<ds>/<system>/ (same VMs, cells, seeds, rounds, root and --log).
sr_bin does not read g0 and comes from B1 in both modes.

Contrasts (per seed, mean over the 19 cells; CI with t over 5 seeds):
  1. S@par - S@nat for S in tri_all, cos_bin, cosserver_only
  2. tri_all@par - best single signal under parity (best by overall mean among cos_bin@par,
     cosserver_only@par, sr_bin, per dataset and H)
  3. tri_all@nat - best native single signal
Rule: H1 (BloodMNIST, H = 250) at risk if the CI95 of 2 is not entirely above 0 or its mean is below
half of 3. H4 (MNIST, H = 150) at risk if the lower CI90 bound of tri_all@par - cos_bin@par < -1 p.p.

Usage: venv/bin/python scripts/analyze_step_parity.py
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

PAR = "results/tri_all_step_parity"
NAT = "results/inclusion_mechanism"
FLT = "results/fltrust_parity"
OUT = f"{PAR}/analysis"
SEEDS = [72, 73, 74, 75, 76]
GSIG = ["tri_all", "cos_bin", "cosserver_only"]
HORIZONS = {"bloodmnist": [250, 300], "mnist": [150]}
KEY = ["alpha", "attack_type", "seed"]
N_CELLS = 19


def load(base, ds, system, tag=""):
    fs = [f for f in glob.glob(f"{base}/raw/{ds}/{system}{tag}/*.csv") if not f.endswith(".log.csv")]
    d = pd.concat([pd.read_csv(f, usecols=KEY + ["round", "accuracy"]) for f in fs], ignore_index=True)
    return d[d.seed.isin(SEEDS)]


def at(d, h):
    x = d[d["round"] == h].set_index(KEY)["accuracy"]
    assert len(x) == N_CELLS * len(SEEDS), (h, len(x))
    return x


def ci(diff_by_seed, level):
    n = len(diff_by_seed)
    m, se = diff_by_seed.mean(), diff_by_seed.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return m, m - t * se, m + t * se


def contrast(a, b, level=0.95):
    d = (100 * (a - b)).groupby(level="seed").mean()
    m, lo, hi = ci(d.values, level)
    return {"mean": m, "lo": lo, "hi": hi, "seeds_pos": int((d > 0).sum())}


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, verdict = [], []
    for ds, hs in HORIZONS.items():
        par = {s: load(PAR, ds, s, "@stmedian") for s in GSIG}
        nat = {s: load(NAT, ds, s) for s in GSIG + ["sr_bin"]}
        for h in hs:
            P = {s: at(par[s], h) for s in GSIG}
            N = {s: at(nat[s], h) for s in GSIG + ["sr_bin"]}
            P["sr_bin"] = N["sr_bin"]
            for s in GSIG:
                rows.append({"dataset": ds, "H": h, "contrast": f"{s}@par - {s}@nat", **contrast(P[s], N[s])})
            sigs = ["cos_bin", "cosserver_only", "sr_bin"]
            best_p = max(sigs, key=lambda s: P[s].mean())
            best_n = max(sigs, key=lambda s: N[s].mean())
            c2 = contrast(P["tri_all"], P[best_p])
            c3 = contrast(N["tri_all"], N[best_n])
            rows.append({"dataset": ds, "H": h, "contrast": f"tri_all@par - best@par ({best_p})", **c2})
            rows.append({"dataset": ds, "H": h, "contrast": f"tri_all@nat - best@nat ({best_n})", **c3})
            for s in sigs:
                rows.append({"dataset": ds, "H": h, "contrast": f"tri_all@par - {s}{'' if s == 'sr_bin' else '@par'}",
                             **contrast(P["tri_all"], P[s])})
            for a in sorted(P["tri_all"].index.get_level_values("alpha").unique()):
                sel = lambda x: x[x.index.get_level_values("alpha") == a]
                rows.append({"dataset": ds, "H": h, "contrast": f"alpha={a}: tri_all@par - {best_p}@par",
                             **contrast(sel(P["tri_all"]), sel(P[best_p]))})
            if ds == "bloodmnist" and h == 250:
                risk = (c2["lo"] <= 0) or (c2["mean"] < c3["mean"] / 2)
                verdict.append(f"H1 (BloodMNIST, H=250): contrast 2 = {c2['mean']:+.2f} [{c2['lo']:+.2f}; {c2['hi']:+.2f}] "
                               f"(best@par {best_p}), contrast 3 = {c3['mean']:+.2f} (best@nat {best_n}) -> "
                               + ("AT RISK" if risk else "not at risk"))
            if ds == "mnist":
                c4 = contrast(P["tri_all"], P["cos_bin"], level=0.90)
                rows.append({"dataset": ds, "H": h, "contrast": "tri_all@par - cos_bin@par (CI90)", **c4})
                risk = c4["lo"] < -1.0
                verdict.append(f"H4 (MNIST, H=150): tri_all@par - cos_bin@par = {c4['mean']:+.2f}, CI90 "
                               f"[{c4['lo']:+.2f}; {c4['hi']:+.2f}] -> " + ("AT RISK" if risk else "not at risk"))
            flt = glob.glob(f"{FLT}/raw/{ds}/fltrust@stmedian/*.csv")
            if flt:
                f = pd.concat([pd.read_csv(x, usecols=KEY + ["round", "accuracy"]) for x in flt])
                f = f[f.seed.isin(SEEDS)]
                if (f["round"] == h).sum() == N_CELLS * len(SEEDS):
                    rows.append({"dataset": ds, "H": h, "contrast": "tri_all@par - fltrust@par (cross-machine)",
                                 **contrast(P["tri_all"], at(f, h))})
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/contrasts.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(2).to_string(index=False))
    print("\n".join(verdict))
    with open(f"{OUT}/verdict.txt", "w") as fh:
        fh.write("\n".join(verdict) + "\n")


if __name__ == "__main__":
    main()
