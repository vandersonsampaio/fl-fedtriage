# Result — confirmation of TRIAGE (`tri_all`)

**Date:** 2026-10-23.
**Preregistration:** `PREREGISTRATION.md`, frozen on 2026-10-19 09:06:20 -0300. Before the analysis, the hashes in `PREREGISTRATION.sha256` were checked and all seven files matched.

**Verdict: confirmed.** H1 and H2 are significant after Holm on BloodMNIST, and on MNIST `tri_all` is non-inferior to the best single signal. The literal TOST of H4 does not pass, because `tri_all` may be better than the best single signal by more than the margin (§3).

## 1. Run

- **Grid:** 6,140 jobs from 2026-10-19 09:10 to 2026-10-23 ~10:00, with 14 parallel jobs on CPU.
  - 13 systems × 19 cells × seeds 62–71 on each dataset, with BloodMNIST run for 300 rounds and MNIST for 150. The no-attack ceiling has 3 cells per dataset.
  - 4 root-dependent systems with `@r20` on both datasets.
- **Health:** every job wrote its accuracy and there was no traceback. No job was rerun.
- **Reproducibility:** 8 jobs were run again from scratch and compared with the grid on every round they share. They cover both datasets, both roots and 7 systems, and the run lengths were 40 rounds, H = 150 or H = 300. The result was |Δ| = 0 on all of them.

```
venv/bin/python -m scripts.triage_compare --dataset bloodmnist --raw results/f3_confirmation/raw \
    --candidates tri_all --single cosserver_only cos_bin sr_bin \
    --refs fltrust flame fedavg trimmed_mean median clustering krum oracle_uniform \
    --seeds 62 63 64 65 66 67 68 69 70 71 --horizons 150 250 300 --out results/f3_confirmation
venv/bin/python -m scripts.triage_compare --dataset mnist ... --horizons 150 --out results/f3_confirmation
venv/bin/python -m scripts.triage_compare --dataset bloodmnist --raw results/f3_confirmation/raw \
    --candidates tri_all@r20 --single cosserver_only@r20 cos_bin@r20 sr_bin --refs fltrust@r20 flame \
    --seeds 62 63 64 65 66 67 68 69 70 71 --horizons 150 250 300 --out results/f3_confirmation/r20
venv/bin/python -m scripts.triage_compare --dataset mnist ... --horizons 150 --out results/f3_confirmation/r20
venv/bin/python -m scripts.triage_confirm --raw results/f3_confirmation/raw --out results/f3_confirmation
```

**Deviations from the written procedure.** None of them changes the data, the configuration or the set of systems.
1. **One-sided tests.** `scripts.triage_compare` reports a two-sided Wilcoxon p. The one-sided Wilcoxon and the Holm correction that the preregistration specifies come from `scripts/triage_confirm.py`, which was written after the run. It reads the same per-seed means, and it imports the loader of `triage_compare`.
2. **Ceiling outside `--refs`.** `ceiling_fedavg10` runs without attack, so it has no rows in the 19 attack cells. Passing it in `--refs` empties the paired table, so it is reported per α in §5.

## 2. Primary hypotheses (BloodMNIST, H = 250, canonical root)

The best single signal on seeds 62–71 was `cos_bin`, chosen by overall mean (`cos_bin` 0.613, `cosserver_only` 0.605, `sr_bin` 0.558).

| | reference | `tri_all` | reference acc. | diff (p.p.) | seeds > 0 | p (one-sided) | p (Holm) |
|---|---|---:|---:|---:|---:|---:|---:|
| **H1** | best single signal (`cos_bin`) | 0.647 | 0.613 | **+3.36** | 10/10 | 0.0010 | **0.0029** |
| **H2** | `fltrust` | 0.647 | 0.619 | **+2.74** | 9/10 | 0.0020 | **0.0029** |
| **H3** | `flame` | 0.647 | 0.578 | **+6.89** | 10/10 | 0.0010 | **0.0029** |

All three are significant. For comparison with development (seeds 72–81, H = 250): H1 was +3.62, H2 +5.37 and H3 +6.00. The H2 gain shrank by half on the new seeds.

**Per cell against `cos_bin`** (beyond 2 SE):
- 10 wins, 6 ties, 3 losses.
- The three losses are `sign_flipping` at α = 0.1 (−2.9), and `label_flipping` (−3.6) and `sign_flipping` (−3.3) at α = 0.5.

**Other horizons and references** (`compare_bloodmnist.txt`):

| H | vs best single | vs `fltrust` | vs `flame` | vs per-cell signal oracle | vs `oracle_uniform` |
|---|---:|---:|---:|---:|---:|
| 150 | +5.03 | +4.14 | +6.91 | +2.04 | −1.19 |
| 250 | +3.36 | +2.74 | +6.89 | +1.03 | −1.18 |
| 300 | +3.66 | +3.47 | +6.06 | +1.16 | −1.99 |

## 3. H4 (MNIST, H = 150, canonical root)

The best single signal was `cos_bin`.

| `tri_all` | `cos_bin` | diff (p.p.) | 90% CI | p (diff > −1) | p (diff < +1) | TOST p |
|---:|---:|---:|---|---:|---:|---:|
| 0.885 | 0.876 | +0.92 | [+0.66, +1.17] | < 0.0001 | 0.28 | 0.28 |

- **The hypothesis holds as stated.** It says that `tri_all` is not inferior to the best single signal. The lower test of the TOST rejects a loss of 1 p.p. or more, and every seed is in favor of `tri_all` (two-sided Wilcoxon p = 0.002).
- **Equivalence within ±1 p.p. is not shown.** The upper end of the interval is above +1 p.p., which means `tri_all` may be better than `cos_bin` by more than the margin.
- **Why the two readings differ.** The preregistration names the hypothesis as non-inferiority (§4) but names the procedure as a TOST (§5). This mismatch was there before the run. The verdict above follows the hypothesis, and this table reports both readings.

Per cell against `cos_bin`: 7 wins, 8 ties, 4 losses. The losses are `label_flipping` at all three α (−2.3 to −5.7) and `gaussian_noise` at α = 0.05 (−2.0).

## 4. Secondary hypotheses (BloodMNIST, H = 250, root `@r20`)

The best single signal on the same root was `cos_bin@r20` (0.566; `sr_bin` 0.558, `cosserver_only@r20` 0.551).

| | reference | `tri_all@r20` | diff (p.p.) | seeds > 0 | p (one-sided) | p (Holm) |
|---|---|---:|---:|---:|---:|---:|
| **H1′** | `cos_bin@r20` | 0.624 | **+5.73** | 10/10 | 0.0010 | **0.0029** |
| **H2′** | `fltrust@r20` | 0.624 | **+2.07** | 8/10 | 0.0098 | **0.0098** |
| **H3′** | `flame` | 0.624 | **+4.57** | 10/10 | 0.0010 | **0.0029** |

All three are significant. On MNIST, `tri_all@r20` is +2.20 p.p. over the best single signal on the same root (`sr_bin`) and +9.84 over `fltrust@r20` (descriptive).

## 5. Descriptive results

**Root cost** (`confirm_root_cost.csv`): each system with `@r20` minus the same system with the canonical root, in p.p.

| system | BloodMNIST H = 150 | H = 250 | H = 300 | MNIST H = 150 |
|---|---:|---:|---:|---:|
| `tri_all` | −0.69 | **−2.32** | −1.79 | −0.98 |
| `cos_bin` | −3.12 | −4.69 | −2.57 | −4.06 |
| `cosserver_only` | −6.12 | −5.34 | −4.33 | −3.36 |
| `fltrust` | −1.80 | −1.64 | +0.17 | −9.45 |

- At H = 150 on both datasets, `tri_all` loses the least to the smaller root, as in development (−0.95 on BloodMNIST and −0.90 on MNIST, `results/root_cost`).
- On BloodMNIST at H = 250 and H = 300, `fltrust` loses less than `tri_all`.
- On BloodMNIST at H = 250 the loss of `tri_all` is 2.32 p.p., above the 2 p.p. target for a 20-sample root.

**Worst case** (`confirm_worst_case.csv`): `tri_all` minus the best fixed rule of each cell.
- On BloodMNIST the best fixed rule of a cell is often plain `fedavg`.
- The worst cells at H = 250 are `fltrust_aligned` (−5.6), `sign_flipping` (−5.5) and `label_flipping` (−4.9) at α = 0.5.
- At H = 300, `sign_flipping` at α = 0.05 is −17.0. This is the late collapse already documented in `results/collapse_diag`: the same cell is +5.2 over `cos_bin` at H = 250 and −8.5 at H = 300.
- On MNIST the worst cells are `label_flipping`: −5.7 at α = 0.05, −3.5 at α = 0.1 and −3.2 at α = 0.5.

**No-attack ceiling** (`confirm_ceiling.csv`): `tri_all` under attack against `ceiling_fedavg10` without attack.

| dataset, H | α = 0.05 | α = 0.1 | α = 0.5 |
|---|---:|---:|---:|
| BloodMNIST, 250 | +1.03 | −8.94 | −3.08 |
| MNIST, 150 | −3.69 | −0.56 | −1.34 |

## 6. Conclusion

- **Primary:** `tri_all` is confirmed on seeds that were never used in development.
  - On BloodMNIST at H = 250 it beats the best single signal by 3.4 p.p., FLTrust by 2.7 p.p. and FLAME by 6.9 p.p.
  - On MNIST at H = 150 it is not inferior to the best single signal; the point estimate is +0.9 p.p.
- **Secondary:** the hypotheses also hold with a 20-sample root.
- **Limitations seen again:**
  - `label_flipping` on MNIST.
  - `sign_flipping` at α = 0.05 after H = 250.
  - On BloodMNIST, the cells at α = 0.5 where plain averaging is better.
  - A root cost slightly above the 2 p.p. target on BloodMNIST at H = 250.
- **Out of scope:** the learned defenses, adaptive attacks and DP on the server update are evaluated separately, with their own preregistration.

## Files

| file | content |
|---|---|
| `raw/` | per-round accuracy of every job |
| `compare_{bloodmnist,mnist}.{txt,csv}`, `*_cells.csv` | `triage_compare` output, canonical root |
| `r20/` | the same for root `@r20` |
| `confirm.txt`, `confirm_tests.csv`, `confirm_h4.csv` | preregistered tests |
| `confirm_root_cost.csv`, `confirm_worst_case.csv`, `confirm_ceiling.csv` | descriptive results |
