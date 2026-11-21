# Longer horizon on BloodMNIST (H = 300)

At H = 150 nothing has converged on BloodMNIST, not even FedAvg without attack, and between rounds
100 and 150 `cos_bin` and `fltrust` gain about twice as much as `tri_all`. This experiment reruns the
main systems for 300 rounds to check whether the advantage of `tri_all` holds once the curves level
off, or is only faster convergence.

BloodMNIST, logistic model, 10 clients (0 and 1 Byzantine), the 19 attack cells, seeds 72–76,
300 rounds; `tri_all`, `sr_bin`, `cos_bin`, `cosserver_only`, `fltrust`, `fedavg`, `oracle_uniform`
and `ceiling_fedavg10` (680 runs). Rounds 1–150 reproduce the H = 150 runs of
`results/bloodmnist_reference/` exactly.

```
venv/bin/python -m scripts.triage_grid jobs --datasets bloodmnist \
    --systems tri_all sr_bin cos_bin cosserver_only fltrust fedavg oracle_uniform ceiling_fedavg10 \
    --seeds 72 73 74 75 76 --rounds 300 --out results/horizon300 > jobs.txt
bash scripts/launch_grid.sh horizon300 jobs.txt -P 14
venv/bin/python -m scripts.triage_compare --dataset bloodmnist --raw results/horizon300/raw \
    --candidates tri_all --single cosserver_only cos_bin sr_bin --refs fltrust fedavg oracle_uniform \
    --horizons 150 250 300 --out results/horizon300
```

## Convergence

| system | H = 150 | H = 200 | H = 250 | H = 300 |
|---|---:|---:|---:|---:|
| `ceiling_fedavg10` (no attack) | 0.672 | 0.673 | 0.688 | 0.683 |
| `oracle_uniform` | 0.644 | 0.647 | 0.667 | 0.667 |
| `tri_all` | 0.634 | 0.636 | 0.650 | 0.646 |
| `cosserver_only` | 0.598 | 0.603 | 0.614 | 0.620 |
| `cos_bin` | 0.600 | 0.601 | 0.623 | 0.609 |
| `fltrust` | 0.579 | 0.595 | 0.596 | 0.611 |
| `sr_bin` | 0.562 | 0.566 | 0.565 | 0.566 |
| `fedavg` | 0.554 | 0.563 | 0.570 | 0.570 |

The ceiling levels off around round 250.

## `tri_all` against each system (p.p., paired by cell and seed)

| reference | H = 150 | H = 250 | H = 300 |
|---|---:|---:|---:|
| `cosserver_only` | +3.57 (5/5) | +3.62 (5/5) | +2.57 (4/5) |
| `cos_bin` | +3.41 (5/5) | +2.62 (5/5) | +3.68 (5/5) |
| `fltrust` | +5.42 (5/5) | +5.37 (5/5) | +3.46 (4/5) |
| `sr_bin` | +7.16 (5/5) | +8.49 (5/5) | +7.99 (5/5) |
| `fedavg` | +7.93 (5/5) | +8.00 (5/5) | +7.59 (5/5) |
| per-cell oracle over the single signals | +0.58 | +0.25 | +0.52 |
| `oracle_uniform` | −1.03 | −1.72 | −2.12 |

In parentheses, seeds with `tri_all` ahead (Wilcoxon p = 0.0625 for 5/5, 0.125 for 4/5). The full
tables, with TOST and the per-cell results, are in `compare_bloodmnist.txt` and the CSVs next to it.

## Findings

- **The advantage holds up to round 250.** At H = 250, where the curves level off, `tri_all` is
  still 3.6 p.p. above the best single signal and 5.4 p.p. above `fltrust`, with every seed ahead.
  It is not only faster convergence.
- **`tri_all` collapses late on sign flipping at α = 0.05.** Its accuracy stays near 0.65 until
  round 200 and falls to about 0.49 at round 300 on all five seeds, while the honest-only oracle
  rises to 0.70. This cell alone (−13.8 p.p. against the best single signal, −21 p.p. against the
  oracle) accounts for most of the drop of the advantage at H = 300, where the gain over
  `cosserver_only` and `fltrust` falls to 4 of 5 seeds. Smaller late drops (11–13 p.p. on some
  seeds) appear at α = 0.1 on label flipping and `low_mag_backdoor`. These runs do not log the
  inclusion decisions, so the cause is not checked.
- **`tri_all` stays level with the per-cell oracle over the single signals** (+0.25 to +0.58 p.p.,
  not significant) and below the honest-only oracle, with a gap that grows from 1.0 to 2.1 p.p.
