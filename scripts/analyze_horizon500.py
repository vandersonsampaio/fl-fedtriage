"""
Late collapse of tri_all up to 500 rounds (block D1), BloodMNIST, seeds 72-76. Plan, definitions and
reading rule: results/horizon500/PLAN.md (written before the runs). Writes results/horizon500/analysis/.

Definitions (PLAN.md):
  MA10(r)  = mean accuracy over rounds r-9..r of one run
  collapse = some round r >= 60 with MA10(r) <= running max of MA10 up to r - 5 p.p.
  onset    = first such round; late collapse = onset > 250
Rule: a cell "has a late collapse" for tri_all if >= 2/5 seeds show one; it is attributed to the AND
rule only if cos_bin has a collapse (any onset, the term defined above) in fewer seeds in the same cell
AND ceiling_fedavg10 (no attack) at the same alpha collapses in no seed. The comparison with cos_bin's
late collapses only is also reported, as a non-plan reading (an early collapse precludes a late one). For those cells, the inclusion logs are read by window of 50
rounds (Byzantine inclusion rate, honest exclusion rate, norm ratio of included Byzantine / honest).

Determinism check (PLAN.md): rounds 1-300 of tri_all and cos_bin must equal the B1 runs
(results/inclusion_mechanism/, all made on the same VMs) with |delta| = 0.

Usage: venv/bin/python scripts/analyze_horizon500.py
"""

import glob
import os

import numpy as np
import pandas as pd

BASE = "results/horizon500"
B1 = "results/inclusion_mechanism"
OUT = f"{BASE}/analysis"
SEEDS = [72, 73, 74, 75, 76]
KEY = ["alpha", "attack_type", "seed"]
CELL = ["alpha", "attack_type"]
DROP = 0.05
MIN_ROUND = 60
LATE = 250
WINDOWS = list(range(0, 501, 50))


def read_acc(base, system):
    fs = [f for f in glob.glob(f"{base}/raw/bloodmnist/{system}/*.csv") if not f.endswith(".log.csv")]
    return pd.concat([pd.read_csv(f, usecols=["system", "alpha", "attack_type", "seed", "round", "accuracy"])
                      for f in fs], ignore_index=True)


def collapse(acc):
    """acc: accuracies of one run ordered by round (index = round). Returns (onset or nan, MA10 series)."""
    ma = acc.rolling(10).mean()
    runmax = ma.cummax()
    hit = ma[(ma.index >= MIN_ROUND) & (ma <= runmax - DROP + 1e-12)]
    return (float(hit.index[0]) if len(hit) else np.nan), ma


def per_run(d):
    rows = []
    for (sys_, a, att, s), g in d.groupby(["system"] + KEY):
        g = g.set_index("round")["accuracy"].sort_index()
        assert len(g) == 500 and g.index[-1] == 500, (sys_, a, att, s)
        onset, ma = collapse(g)
        rows.append({"system": sys_, "alpha": a, "attack_type": att, "seed": s, "onset": onset,
                     "late": bool(onset > LATE) if not np.isnan(onset) else False,
                     "collapsed_by_300": bool(onset <= 300) if not np.isnan(onset) else False,
                     "collapsed_by_400": bool(onset <= 400) if not np.isnan(onset) else False,
                     "collapsed_by_500": not np.isnan(onset),
                     "peak_ma10": ma.max(), "final_ma10": ma.iloc[-1],
                     "final_minus_peak_pp": 100 * (ma.iloc[-1] - ma.max()),
                     **{f"acc_{h}": g.loc[h] for h in (250, 300, 400, 500)}})
    return pd.DataFrame(rows)


def per_cell(r):
    agg = r.groupby(["system"] + CELL).agg(
        n_by_300=("collapsed_by_300", "sum"), n_by_400=("collapsed_by_400", "sum"),
        n_by_500=("collapsed_by_500", "sum"), n_late=("late", "sum"),
        median_onset=("onset", "median"), mean_final_minus_peak_pp=("final_minus_peak_pp", "mean"),
        acc_250=("acc_250", "mean"), acc_300=("acc_300", "mean"), acc_400=("acc_400", "mean"),
        acc_500=("acc_500", "mean"), n=("seed", "count")).reset_index()
    assert (agg.n == 5).all()
    return agg


def determinism():
    out = []
    for s in ["tri_all", "cos_bin"]:
        new = read_acc(BASE, s)
        old = read_acc(B1, s)
        m = new[new["round"] <= 300].merge(old, on=KEY + ["round"], suffixes=("_new", "_b1"))
        assert len(m) == 95 * 300, (s, len(m))
        dmax = (m.accuracy_new - m.accuracy_b1).abs().max()
        nruns = m.assign(d=(m.accuracy_new - m.accuracy_b1).abs()).groupby(KEY).d.max()
        out.append(f"{s}: {len(m)} rounds compared over {nruns.size} runs; max |delta| = {dmax:.3g}; "
                   f"runs with any difference = {(nruns > 0).sum()}")
        # inclusion decisions, too
        diffs = 0
        for f in glob.glob(f"{BASE}/raw/bloodmnist/{s}/*.log.csv.gz"):
            a = pd.read_csv(f, usecols=["round", "client", "included"])
            b = pd.read_csv(f.replace(BASE, B1), usecols=["round", "client", "included"])
            a = a[a["round"] <= 300].reset_index(drop=True)
            diffs += int((a.values != b.values).any())
        out.append(f"{s}: inclusion logs differing in rounds 1-300 = {diffs} of 95")
    return out


def parse(path):
    name = os.path.basename(path).removesuffix(".log.csv.gz")
    a, rest = name.split("_", 1)
    attack, seed = rest.rsplit("_seed", 1)
    return float(a[1:]), attack, int(seed)


def window_rates(system, alpha, attack, onsets):
    rows = []
    for s in SEEDS:
        f = f"{BASE}/raw/bloodmnist/{system}/a{alpha}_{attack}_seed{s}.log.csv.gz"
        x = pd.read_csv(f, usecols=["round", "client", "included", "is_byz", "neg_norm"])
        x["norm"] = -x["neg_norm"]
        x["window"] = pd.cut(x["round"], WINDOWS)
        for w, g in x.groupby("window", observed=True):
            byz, hon = g[g.is_byz], g[~g.is_byz]
            ib = byz[byz.included]
            rows.append({"system": system, "alpha": alpha, "attack_type": attack, "seed": s, "window": str(w),
                         "onset": onsets.get(s, np.nan),
                         "byz_included": byz.included.mean() if len(byz) else np.nan,
                         "honest_excluded": 1 - hon.included.mean(),
                         "norm_ratio_inc_byz_over_honest": ib.norm.mean() / hon.norm.mean() if len(ib) else np.nan})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    out = ["# D1 analysis (generated by scripts/analyze_horizon500.py)", "", "## Determinism vs B1 (rounds 1-300)"]
    out += determinism()
    d = pd.concat([read_acc(BASE, s) for s in ["tri_all", "cos_bin", "ceiling_fedavg10"]], ignore_index=True)
    r = per_run(d)
    c = per_cell(r)
    r.to_csv(f"{OUT}/runs.csv", index=False)
    c.to_csv(f"{OUT}/cells.csv", index=False)

    out += ["", "## Runs with a collapse (any onset), all cells"]
    out.append(r[r.system != "ceiling_fedavg10"].groupby("system")[["collapsed_by_500", "late"]].sum().to_string())
    out += ["", "## Overall mean accuracy by horizon"]
    out.append(r.groupby("system")[["acc_250", "acc_300", "acc_400", "acc_500"]].mean().round(4).to_string())
    out += ["", "## Collapses per cell (seeds of 5): by 300 / 400 / 500, late (onset > 250), median onset"]
    show = c.pivot_table(index=CELL, columns="system",
                         values=["n_by_300", "n_by_400", "n_by_500", "n_late", "median_onset"])
    out.append(show.to_string())
    out += ["", "## Mean (final MA10 - peak MA10), p.p., and accuracy at 250/500 per cell"]
    out.append(c.pivot_table(index=CELL, columns="system",
                             values=["mean_final_minus_peak_pp", "acc_250", "acc_500"]).round(3).to_string())

    ceil = c[c.system == "ceiling_fedavg10"].set_index("alpha")
    tri = c[c.system == "tri_all"].set_index(CELL)
    cos = c[c.system == "cos_bin"].set_index(CELL)
    out += ["", "## Reading rule (PLAN.md)",
            "ceiling_fedavg10 collapses (any onset) by alpha: "
            + ", ".join(f"{a} {int(ceil.loc[a, 'n_by_500'])}/5" for a in ceil.index)]
    flagged, rate_frames = [], []
    for cell, row in tri.iterrows():
        if row.n_late >= 2:
            a, att = cell
            cos_any = int(cos.loc[cell, "n_by_500"])
            cos_late = int(cos.loc[cell, "n_late"])
            ceil_n = int(ceil.loc[a, "n_by_500"])
            flagged.append({"alpha": a, "attack_type": att, "tri_late": int(row.n_late),
                            "cos_bin_any": cos_any, "cos_bin_late": cos_late, "ceiling_any": ceil_n,
                            "attributed_to_AND_plan": cos_any < row.n_late and ceil_n == 0,
                            "attributed_if_late_only": cos_late < row.n_late and ceil_n == 0,
                            "tri_median_onset": row.median_onset})
            for s in ["tri_all", "cos_bin"]:
                on = r[(r.system == s) & (r.alpha == a) & (r.attack_type == att)].set_index("seed").onset.to_dict()
                rate_frames.append(window_rates(s, a, att, on))
    fl = pd.DataFrame(flagged)
    fl.to_csv(f"{OUT}/late_collapse_cells.csv", index=False)
    out.append(f"Cells with a late collapse for tri_all (>= 2/5 seeds): {len(fl)}")
    if len(fl):
        out.append(fl.to_string(index=False))
        for _, f in fl.iterrows():
            sub = r[(r.alpha == f.alpha) & (r.attack_type == f.attack_type)]
            out.append(f"{f.alpha} {f.attack_type}: per seed onset / final-peak (p.p.) / acc_500")
            out.append(sub[["system", "seed", "onset", "final_minus_peak_pp", "acc_250", "acc_500"]]
                       .round(3).to_string(index=False))
        rt = pd.concat(rate_frames, ignore_index=True)
        rt.to_csv(f"{OUT}/window_rates.csv", index=False)
        out += ["", "## Inclusion by window of 50 rounds in flagged cells (mean over seeds)"]
        out.append(rt.groupby(["alpha", "attack_type", "system", "window"], observed=True, sort=False)
                   [["byz_included", "honest_excluded", "norm_ratio_inc_byz_over_honest"]].mean()
                   .round(3).to_string())
    text = "\n".join(out)
    with open(f"{OUT}/analysis.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
