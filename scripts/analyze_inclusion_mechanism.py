"""
Inclusion log and mechanism of tri_all (block B1), seeds 72-76. Plan and reading rule:
results/inclusion_mechanism/PLAN.md. Writes results/inclusion_mechanism/analysis/.

Accuracy is read at the horizon H (MNIST 150; BloodMNIST 250, the confirmation's horizon, and 300,
the end of the run). The best single signal is chosen per dataset and H by the overall mean over the
19 cells and 5 seeds of this grid, among cos_bin, cosserver_only and sr_bin.

Decomposition (PLAN.md): tri_all's run and the best single signal's run are paired by
(alpha, attack, seed, round, client) over rounds 1..H. The two runs follow different trajectories,
so this is a cross-run, descriptive comparison:
  honest preserved  = honest, included by tri_all, excluded by the signal
  Byzantine excluded = Byzantine, excluded by tri_all, included by the signal
  (and the reverse flows: honest lost, Byzantine let in, reported alongside)
Rule: share = honest preserved / (honest preserved + Byzantine excluded) on BloodMNIST; above 0.5
-> "agreement as protection against false positives", otherwise "agreement as complementary detection".

Usage: venv/bin/python scripts/analyze_inclusion_mechanism.py
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

BASE = "results/inclusion_mechanism"
OUT = f"{BASE}/analysis"
SEEDS = [72, 73, 74, 75, 76]
SIGNALS = ["cos_bin", "cosserver_only", "sr_bin"]
SYSTEMS = ["tri_all"] + SIGNALS
HORIZONS = {"mnist": [150], "bloodmnist": [250, 300]}
WINDOWS = {"mnist": [0, 50, 100, 150], "bloodmnist": [0, 50, 100, 150, 200, 250, 300]}
KEY = ["alpha", "attack_type", "seed"]
LOGKEY = ["alpha", "attack_type", "seed", "round", "client"]


def parse(path):
    name = os.path.basename(path).removesuffix(".log.csv.gz")  # a0.5_sign_flipping_seed72
    a, rest = name.split("_", 1)
    attack, seed = rest.rsplit("_seed", 1)
    return float(a[1:]), attack, int(seed)


def accuracy(ds):
    frames = []
    for s in SYSTEMS:
        for f in glob.glob(f"{BASE}/raw/{ds}/{s}/*.csv"):
            frames.append(pd.read_csv(f, usecols=["system", "alpha", "attack_type", "seed", "round", "accuracy"]))
    d = pd.concat(frames, ignore_index=True)
    assert len(d.groupby(["system"] + KEY)) == 4 * 95, ds
    return d


def logs(ds, system):
    frames = []
    for f in sorted(glob.glob(f"{BASE}/raw/{ds}/{system}/*.log.csv.gz")):
        a, attack, seed = parse(f)
        x = pd.read_csv(f, usecols=["round", "client", "included", "is_byz", "neg_norm"]
                        + (["inversion_flag", "server_cosine_flag"] if system == "tri_all" else []))
        frames.append(x.assign(alpha=a, attack_type=attack, seed=seed))
    d = pd.concat(frames, ignore_index=True)
    assert d.groupby(KEY).ngroups == 95, (ds, system)
    d["norm"] = -d["neg_norm"]
    return d


def ci(per_seed):
    x = np.asarray(per_seed)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    q = stats.t.ppf(0.975, len(x) - 1)
    return m, m - q * se, m + q * se


def rates(d, windows):
    d = d.copy()
    d["window"] = pd.cut(d["round"], windows)
    rows = []
    for (a, att, w), g in d.groupby(["alpha", "attack_type", "window"], observed=True):
        byz, hon = g[g.is_byz], g[~g.is_byz]
        inc_byz = byz[byz.included]
        rows.append({"alpha": a, "attack_type": att, "window": str(w),
                     "byz_included": byz.included.mean() if len(byz) else np.nan,
                     "honest_excluded": 1 - hon.included.mean(),
                     "norm_ratio_inc_byz_over_honest": (inc_byz.norm.mean() / hon.norm.mean()) if len(inc_byz) else np.nan})
    return pd.DataFrame(rows)


def flows(tri, sig, H):
    m = tri[tri["round"] <= H].merge(sig[sig["round"] <= H][LOGKEY + ["included", "is_byz"]],
                                     on=LOGKEY, suffixes=("_tri", "_sig"))
    assert (m.is_byz_tri == m.is_byz_sig).all()
    hon, byz = ~m.is_byz_tri, m.is_byz_tri
    m["honest_preserved"] = hon & m.included_tri & ~m.included_sig
    m["byz_excluded"] = byz & ~m.included_tri & m.included_sig
    m["honest_lost"] = hon & ~m.included_tri & m.included_sig
    m["byz_let_in"] = byz & m.included_tri & ~m.included_sig
    return m


FLOWS = ["honest_preserved", "byz_excluded", "honest_lost", "byz_let_in"]


def share(s):
    hp, be = s["honest_preserved"], s["byz_excluded"]
    return hp / (hp + be) if hp + be else np.nan


def main():
    os.makedirs(OUT, exist_ok=True)
    out, gain_rows, flow_rows, rate_frames, within_rows = [], [], [], [], []
    for ds in ["bloodmnist", "mnist"]:
        acc = accuracy(ds)
        L = {s: logs(ds, s) for s in SYSTEMS}
        for s in SYSTEMS:
            rate_frames.append(rates(L[s], WINDOWS[ds]).assign(dataset=ds, system=s))
        for H in HORIZONS[ds]:
            a = acc[acc["round"] == H].pivot_table(index=KEY, columns="system", values="accuracy")
            means = a.mean()
            best = means[SIGNALS].idxmax()
            out.append(f"## {ds}, H = {H}")
            out.append("Overall means (%): " + ", ".join(f"{s} {100 * means[s]:.2f}" for s in SYSTEMS)
                       + f". Best single signal: {best}.")
            for s in SIGNALS:
                per_seed = (a["tri_all"] - a[s]).groupby(level="seed").mean()
                m, lo, hi = ci(per_seed)
                gain_rows.append({"dataset": ds, "H": H, "signal": s, "is_best": s == best, "gain_pp": 100 * m,
                                  "ci95_lo_pp": 100 * lo, "ci95_hi_pp": 100 * hi,
                                  "per_seed_pp": np.round(100 * per_seed.values, 2).tolist()})
                out.append(f"tri_all - {s}: {100 * m:+.2f} ({100 * lo:+.2f}; {100 * hi:+.2f}) p.p., per seed "
                           f"{np.round(100 * per_seed.values, 2).tolist()}")
            per_alpha = (a["tri_all"] - a[best]).groupby(level="alpha").mean()
            out.append("tri_all - best by alpha (p.p.): " + ", ".join(f"{k} {100 * v:+.2f}" for k, v in per_alpha.items()))

            for s in SIGNALS:
                m = flows(L["tri_all"], L[s], H)
                tot = m[FLOWS].sum()
                rec = {"dataset": ds, "H": H, "signal": s, "is_best": s == best, "scope": "all",
                       **{k: int(v) for k, v in tot.items()}, "share_honest_preserved": share(tot)}
                flow_rows.append(rec)
                for (grp, key) in (("seed", "seed"), ("alpha", "alpha"), ("attack", "attack_type")):
                    for k, g in m.groupby(key):
                        t = g[FLOWS].sum()
                        flow_rows.append({"dataset": ds, "H": H, "signal": s, "is_best": s == best,
                                          "scope": f"{grp}={k}", **{c: int(v) for c, v in t.items()},
                                          "share_honest_preserved": share(t)})
                # cells where tri_all beat the signal at H
                cg = (a["tri_all"] - a[s])
                pos = cg[cg > 0].index
                g = m.set_index(KEY).loc[m.set_index(KEY).index.isin(pos)]
                t = g[FLOWS].sum()
                flow_rows.append({"dataset": ds, "H": H, "signal": s, "is_best": s == best,
                                  "scope": "cells with positive gain", **{c: int(v) for c, v in t.items()},
                                  "share_honest_preserved": share(t)})
                if s == best:
                    seed_shares = [share(g[FLOWS].sum()) for _, g in m.groupby("seed")]
                    out.append(f"Flows vs {s} (rounds 1..{H}): " + ", ".join(f"{k} {int(v)}" for k, v in tot.items()))
                    out.append(f"Share of honest preserved: {share(tot):.3f} (per seed "
                               f"{np.round(seed_shares, 3).tolist()}); net honest {int(tot.honest_preserved - tot.honest_lost)}, "
                               f"net Byzantine excluded {int(tot.byz_excluded - tot.byz_let_in)}")
                    if ds == "bloodmnist":
                        verdict = ("agreement as protection against false positives" if share(tot) > 0.5
                                   else "agreement as complementary detection")
                        out.append(f"Rule (PLAN.md): share {'>' if share(tot) > 0.5 else '<='} 0.5 -> {verdict}")

            # within tri_all's own run: clients flagged by one signal, rescued by the other (combine='all')
            t = L["tri_all"][L["tri_all"]["round"] <= H]
            for flag in ["server_cosine_flag", "inversion_flag"]:
                f = t[t[flag]]
                for byz, g in f.groupby("is_byz"):
                    within_rows.append({"dataset": ds, "H": H, "flag": flag, "is_byz": byz, "n_flagged": len(g),
                                        "share_included_by_tri_all": g.included.mean()})
            out.append("")
    gains = pd.DataFrame(gain_rows)
    fl = pd.DataFrame(flow_rows)
    rt = pd.concat(rate_frames, ignore_index=True)
    wi = pd.DataFrame(within_rows)
    gains.to_csv(f"{OUT}/gains.csv", index=False)
    fl.to_csv(f"{OUT}/flows.csv", index=False)
    rt.to_csv(f"{OUT}/rates.csv", index=False)
    wi.to_csv(f"{OUT}/within_tri_all.csv", index=False)

    out.append("## Rates over all rounds and cells (byz included = FN, honest excluded = FP)")
    for ds in ["bloodmnist", "mnist"]:
        for s in SYSTEMS:
            d = logs(ds, s)
            out.append(f"{ds} {s}: byz included {d[d.is_byz].included.mean():.3f}, "
                       f"honest excluded {1 - d[~d.is_byz].included.mean():.3f}")
    out.append("")
    out.append("## Within tri_all's run: flagged by one signal, still included (the other did not flag)")
    out.append(wi.round(3).to_string(index=False))
    text = "\n".join(out)
    with open(f"{OUT}/analysis.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
