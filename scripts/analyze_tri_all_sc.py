"""
Block D2: tri_all_sc (tri_all plus a cap on included norms above 3 x their median) against tri_all.
Plan and criterion: results/tri_all_sc/PLAN.md. Writes results/tri_all_sc/analysis/.

References (same VMs): BloodMNIST tri_all from D1 (results/horizon500/raw/bloodmnist/tri_all, 500 rounds);
MNIST tri_all from B1 (results/inclusion_mechanism/raw/mnist/tri_all, 150 rounds).

Usage: venv/bin/python scripts/analyze_tri_all_sc.py
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

SC = "results/tri_all_sc"
REF = {"bloodmnist": "results/horizon500/raw/bloodmnist/tri_all",
       "mnist": "results/inclusion_mechanism/raw/mnist/tri_all"}
OUT = f"{SC}/analysis"
SEEDS = [72, 73, 74, 75, 76]
KEY = ["alpha", "attack_type", "seed"]
N_CELLS = 19
TARGET = (0.05, "sign_flipping")


def load(path):
    fs = [f for f in glob.glob(f"{path}/*.csv") if not f.endswith(".log.csv")]
    d = pd.concat([pd.read_csv(f, usecols=KEY + ["round", "accuracy"]) for f in fs], ignore_index=True)
    d = d[d.seed.isin(SEEDS)]
    assert d.groupby(KEY).ngroups == N_CELLS * len(SEEDS), (path, d.groupby(KEY).ngroups)
    return d


def curves(d):
    return {k: g.sort_values("round")["accuracy"].to_numpy() for k, g in d.groupby(KEY)}


def ma10(acc):
    return pd.Series(acc).rolling(10).mean().to_numpy()


def peak_to_end(acc, h):
    m = ma10(acc[:h])
    return 100 * (np.nanmax(m) - m[-1])


def collapses(acc, h):
    m = ma10(acc[:h])
    run = np.fmax.accumulate(np.nan_to_num(m, nan=-np.inf))
    r = np.arange(1, h + 1)
    return bool(((r >= 60) & (m <= run - 0.05)).any())


def ci95(x):
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return m, m - t * se, m + t * se


def main():
    os.makedirs(OUT, exist_ok=True)
    lines, rows = [], []
    sc = {ds: curves(load(f"{SC}/raw/{ds}/tri_all_sc")) for ds in REF}
    ref = {ds: curves(load(REF[ds])) for ds in REF}

    # Criterion 1
    drops = {s: peak_to_end(sc["bloodmnist"][(*TARGET, s)], 500) for s in SEEDS}
    drops_ref = {s: peak_to_end(ref["bloodmnist"][(*TARGET, s)], 500) for s in SEEDS}
    ok1 = sum(v <= 5 for v in drops.values()) >= 4
    lines.append("Criterion 1 (BloodMNIST, sign_flipping a0.05, H=500, peak-to-end drop <= 5 p.p. in >= 4/5 seeds): "
                 + ("PASS" if ok1 else "FAIL"))
    lines.append("  tri_all_sc drops: " + ", ".join(f"{s}: {drops[s]:.1f}" for s in SEEDS))
    lines.append("  tri_all    drops: " + ", ".join(f"{s}: {drops_ref[s]:.1f}" for s in SEEDS))

    # Criterion 2 and descriptive deltas
    ok2 = True
    for ds, hs in (("bloodmnist", [250, 300, 400, 500]), ("mnist", [150])):
        for h in hs:
            delta = {k: 100 * (sc[ds][k][h - 1] - ref[ds][k][h - 1]) for k in sc[ds]}
            d = pd.Series(delta)
            d.index = pd.MultiIndex.from_tuples(d.index, names=KEY)
            by_seed = d.groupby(level="seed").mean()
            by_cell = d.groupby(level=["alpha", "attack_type"]).mean()
            m, lo, hi = ci95(by_seed)
            for (a, att), v in by_cell.items():
                rows.append({"dataset": ds, "H": h, "alpha": a, "attack_type": att, "delta_mean": v})
            rows.append({"dataset": ds, "H": h, "alpha": "all", "attack_type": "all", "delta_mean": m,
                         "lo": lo, "hi": hi, "seeds_neg": int((by_seed < 0).sum())})
            worst = "a{} {}".format(*by_cell.idxmin())
            primary = (ds == "bloodmnist" and h == 250) or ds == "mnist"
            if primary:
                c = m >= -1 and by_cell.min() >= -5
                ok2 &= c
            lines.append(f"{'Criterion 2' if primary else 'Descriptive'} {ds} H={h}: mean delta {m:+.2f} "
                         f"[{lo:+.2f}; {hi:+.2f}], worst cell {worst} {by_cell.min():+.2f}"
                         + (f" -> {'ok' if c else 'FAILS'}" if primary else ""))
            for a in sorted(by_cell.index.get_level_values("alpha").unique()):
                sub = d[d.index.get_level_values("alpha") == a].groupby(level="seed").mean()
                ma, la, ha = ci95(sub)
                lines.append(f"    alpha={a}: {ma:+.2f} [{la:+.2f}; {ha:+.2f}]")
    lines.insert(3, "Criterion 2 (mean delta >= -1 p.p. at Blood H=250 and MNIST H=150, no cell < -5): "
                 + ("PASS" if ok2 else "FAIL"))

    # Criterion 3
    new = []
    cells = sorted({k[:2] for k in sc["bloodmnist"]})
    for cell in cells:
        if cell == TARGET:
            continue
        n_sc = sum(peak_to_end(sc["bloodmnist"][(*cell, s)], 500) > 5 for s in SEEDS)
        n_ref = sum(peak_to_end(ref["bloodmnist"][(*cell, s)], 500) > 5 for s in SEEDS)
        if n_sc >= 2 or n_ref >= 2:
            lines.append(f"  cell {cell}: drop > 5 p.p. in {n_sc}/5 (tri_all_sc), {n_ref}/5 (tri_all)")
        if n_sc >= 2 and n_ref < 2:
            new.append(cell)
    ok3 = not new
    lines.append("Criterion 3 (no new late collapse on BloodMNIST up to H=500): " + ("PASS" if ok3 else f"FAIL {new}"))
    coll = {ds: (sum(collapses(v, len(v)) for v in sc[ds].values()), sum(collapses(v, len(v)) for v in ref[ds].values()))
            for ds in REF}
    lines.append(f"  collapse definition (D1), runs out of 95: {coll}  (tri_all_sc, tri_all)")
    lines.append("VERDICT: " + ("PASS" if ok1 and ok2 and ok3 else "FAIL"))

    # Determinism check and cap rates from the logs
    det, caps = [], []
    for ds in REF:
        for k, acc in sc[ds].items():
            a, att, s = k
            lg = pd.read_csv(f"{SC}/raw/{ds}/tri_all_sc/a{a}_{att}_seed{s}.log.csv.gz")
            capped = lg[lg.included & (lg.cap_scale < 1)]
            first = int(capped["round"].min()) if len(capped) else None
            r_ok = len(acc) if first is None else first - 1
            same = np.array_equal(acc[:r_ok], ref[ds][k][:r_ok])
            det.append({"dataset": ds, "alpha": a, "attack_type": att, "seed": s, "first_cap": first,
                        "rounds_checked": r_ok, "identical": same})
            inc = lg[lg.included]
            for byz, g in inc.groupby("is_byz"):
                for w, gw in g.groupby((g["round"] - 1) // 50):
                    c = gw.cap_scale < 1
                    caps.append({"dataset": ds, "alpha": a, "attack_type": att, "seed": s, "byz": byz,
                                 "window": f"{50 * w + 1}-{50 * w + 50}", "n_included": len(gw),
                                 "frac_capped": c.mean(), "mean_scale_capped": gw.cap_scale[c].mean()})
    det = pd.DataFrame(det)
    det.to_csv(f"{OUT}/determinism.csv", index=False)
    caps = pd.DataFrame(caps)
    caps.to_csv(f"{OUT}/caps_by_window.csv", index=False)
    vac = int((det.first_cap == 1).sum())
    bad = det[(det.rounds_checked > 0) & ~det.identical]
    lines.append(f"Determinism check: {len(det) - vac - len(bad)} runs identical before the first cap, "
                 f"{len(bad)} not identical, {vac} vacuous (first cap in round 1), "
                 f"{int(det.first_cap.isna().sum())} never capped")
    for ds in REF:
        g = caps[caps.dataset == ds]
        for byz in (False, True):
            x = g[g.byz == byz]
            frac = (x.frac_capped * x.n_included).sum() / x.n_included.sum()
            lines.append(f"  {ds} {'Byzantine' if byz else 'honest'} included updates capped: {100 * frac:.2f}%")
        t = g[(g.alpha == TARGET[0]) & (g.attack_type == TARGET[1])] if ds == "bloodmnist" else None
        if t is not None:
            for byz in (False, True):
                x = t[t.byz == byz].groupby("window").apply(
                    lambda z: (z.frac_capped * z.n_included).sum() / z.n_included.sum(), include_groups=False)
                lines.append(f"  target cell, {'Byzantine' if byz else 'honest'} capped by window: "
                             + ", ".join(f"{w} {100 * v:.0f}%" for w, v in sorted(x.items(), key=lambda i: int(i[0].split('-')[0]))))

    pd.DataFrame(rows).to_csv(f"{OUT}/deltas.csv", index=False)
    text = "\n".join(lines)
    print(text)
    with open(f"{OUT}/verdict.txt", "w") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
