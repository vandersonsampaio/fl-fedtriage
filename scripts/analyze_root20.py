"""
tri_all and the cosine signals at root 20, native and under FLTrust step parity, seeds 72-76.
Plan and reading rule: results/tri_all_root20/PLAN.md (amendment: AMENDMENT.md). Writes results/tri_all_root20/analysis/.

Runs: results/tri_all_root20/<native|parity>/raw/<ds>/<system>@r20[@stmedian]/. sr_bin (no g0, no root) comes from
B1, results/inclusion_mechanism/raw/<ds>/sr_bin/. Root-100 references for contrast 4: B1 (native) and
results/tri_all_step_parity/ (parity). FLTrust at root 20 (cross-machine, PC): results/fltrust_parity/raw (parity)
and results/root_cost/raw (native, 150 rounds).

Usage: venv/bin/python scripts/analyze_root20.py
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

R20 = "results/tri_all_root20"
B1 = "results/inclusion_mechanism/raw"
P100 = "results/tri_all_step_parity/raw"
OUT = f"{R20}/analysis"
SEEDS = [72, 73, 74, 75, 76]
GSIG = ["tri_all", "cos_bin", "cosserver_only"]
HORIZONS = {"bloodmnist": [250, 300], "mnist": [150]}
KEY = ["alpha", "attack_type", "seed"]
N_CELLS = 19


def files(path):
    return [f for f in glob.glob(f"{path}/*.csv") if not f.endswith(".log.csv")]


def load(path):
    fs = files(path)
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, usecols=KEY + ["round", "accuracy"]) for f in fs], ignore_index=True)
    return d[d.seed.isin(SEEDS)]


def at(d, h):
    x = d[d["round"] == h].set_index(KEY)["accuracy"]
    return x if len(x) == N_CELLS * len(SEEDS) else None


def ci(v, level):
    n = len(v)
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return m, m - t * se, m + t * se


def contrast(a, b, level=0.95):
    d = (100 * (a - b)).groupby(level="seed").mean()
    m, lo, hi = ci(d.values, level)
    return {"mean": m, "lo": lo, "hi": hi, "seeds_pos": int((d > 0).sum())}


def ma10(acc):
    return pd.Series(acc).rolling(10).mean().to_numpy()


def collapses(acc):
    m = ma10(acc)
    run = np.fmax.accumulate(np.nan_to_num(m, nan=-np.inf))
    r = np.arange(1, len(acc) + 1)
    return bool(((r >= 60) & (m <= run - 0.05)).any())


def inclusion(path):
    """Byzantine included (FN) and honest excluded (FP), pooled over all rounds and runs."""
    inc_b = n_b = exc_h = n_h = 0
    for f in glob.glob(f"{path}/*.log.csv.gz"):
        lg = pd.read_csv(f, usecols=["included", "is_byz"])
        b, h = lg[lg.is_byz], lg[~lg.is_byz]
        inc_b += int(b.included.sum()); n_b += len(b)
        exc_h += int((~h.included).sum()); n_h += len(h)
    return inc_b / n_b, exc_h / n_h


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, verdict, desc = [], [], []
    for ds, hs in HORIZONS.items():
        par = {s: load(f"{R20}/parity/raw/{ds}/{s}@r20@stmedian") for s in GSIG}
        nat = {s: load(f"{R20}/native/raw/{ds}/{s}@r20") for s in GSIG}
        srb = load(f"{B1}/{ds}/sr_bin")
        r100 = {"nat": {s: load(f"{B1}/{ds}/{s}") for s in GSIG},
                "par": {s: load(f"{P100}/{ds}/{s}@stmedian") for s in GSIG}}
        flt = {"par": load(f"results/fltrust_parity/raw/{ds}/fltrust@r20@stmedian"),
               "nat": load(f"results/root_cost/raw/{ds}/fltrust@r20")}
        for h in hs:
            P = {s: at(par[s], h) for s in GSIG}
            N = {s: at(nat[s], h) for s in GSIG}
            assert all(v is not None for v in list(P.values()) + list(N.values())), (ds, h)
            P["sr_bin"] = N["sr_bin"] = at(srb, h)
            row = lambda name, c: rows.append({"dataset": ds, "H": h, "contrast": name, **c})
            for s in GSIG:
                row(f"{s}@par - {s}@nat", contrast(P[s], N[s]))
            sigs = ["cos_bin", "cosserver_only", "sr_bin"]
            best_p = max(sigs, key=lambda s: P[s].mean())
            best_n = max(sigs, key=lambda s: N[s].mean())
            c2, c3 = contrast(P["tri_all"], P[best_p]), contrast(N["tri_all"], N[best_n])
            row(f"tri_all@par - best@par ({best_p})", c2)
            row(f"tri_all@nat - best@nat ({best_n})", c3)
            for s in sigs:
                row(f"tri_all@par - {s}{'' if s == 'sr_bin' else '@par'}", contrast(P["tri_all"], P[s]))
                row(f"tri_all@nat - {s}{'' if s == 'sr_bin' else '@nat'}", contrast(N["tri_all"], N[s]))
            for a in sorted(P["tri_all"].index.get_level_values("alpha").unique()):
                sel = lambda x: x[x.index.get_level_values("alpha") == a]
                row(f"alpha={a}: tri_all@par - {best_p}@par", contrast(sel(P["tri_all"]), sel(P[best_p])))
            for mode, M in (("nat", N), ("par", P)):
                for s in GSIG:
                    ref = at(r100[mode][s], h)
                    if ref is not None:
                        row(f"root sensitivity {s}@{mode}: r20 - r100", contrast(M[s], ref))
                f = at(flt[mode], h) if flt[mode] is not None else None
                if f is not None:
                    row(f"tri_all@{mode} - fltrust@{mode} r20 (cross-machine)", contrast(M["tri_all"], f))
            if ds == "bloodmnist" and h == 250:
                risk = (c2["lo"] <= 0) or (c2["mean"] < c3["mean"] / 2)
                verdict.append(f"H1' side (BloodMNIST, H=250, root 20): contrast 2 = {c2['mean']:+.2f} "
                               f"[{c2['lo']:+.2f}; {c2['hi']:+.2f}] (best@par {best_p}), contrast 3 = {c3['mean']:+.2f} "
                               f"(best@nat {best_n}) -> " + ("AT RISK" if risk else "not at risk"))
            if ds == "mnist":
                c4 = contrast(P["tri_all"], P["cos_bin"], level=0.90)
                row("tri_all@par - cos_bin@par (CI90)", c4)
                verdict.append(f"MNIST side (H=150, root 20): tri_all@par - cos_bin@par = {c4['mean']:+.2f}, CI90 "
                               f"[{c4['lo']:+.2f}; {c4['hi']:+.2f}] -> " + ("AT RISK" if c4["lo"] < -1.0 else "not at risk"))
        for mode, tag in (("native", "@r20"), ("parity", "@r20@stmedian")):
            for s in GSIG:
                fn, fp = inclusion(f"{R20}/{mode}/raw/{ds}/{s}{tag}")
                d = (nat if mode == "native" else par)[s]
                n_coll = sum(collapses(g.sort_values("round").accuracy.to_numpy()) for _, g in d.groupby(KEY))
                desc.append(f"{ds} {s}@{mode[:3]}: Byzantine included {fn:.3f}, honest excluded {fp:.3f}, "
                            f"collapses (D1 definition) {n_coll}/95")
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/contrasts.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.round(2).to_string(index=False))
    text = "\n".join(verdict + [""] + desc)
    print(text)
    with open(f"{OUT}/verdict.txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
