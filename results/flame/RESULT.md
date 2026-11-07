# FLAME and the clipped single signal

Two comparisons set before this run:

- **FLAME** (Nguyen et al., 2022), the closest defense to TRIAGE that needs neither a root dataset
  nor the inversion score: HDBSCAN over the pairwise cosine distances of the updates (minimum
  cluster size n/2 + 1, majority cluster kept), the admitted updates clipped to the median norm,
  and Gaussian noise of standard deviation 0.001 times that median (`flame`), or no noise
  (`flame_nonoise`). If FLAME were level with or above `tri_all` (difference ≤ 0 p.p. or fewer than
  4 of 5 seeds), the contribution of TRIAGE would be auditability, cost and the use of the root,
  not accuracy.
- **`cos_bin_clip`**, the server cosine alone followed by the same norm clipping as `tri_all_clip`,
  to check whether the inversion score pays off once clipping is in place (gain of `tri_all_clip`
  of at least 1 p.p. in at least 4 of 5 seeds).

Logistic model, 10 clients (0 and 1 Byzantine), the 19 attack cells and the trigger backdoor,
seeds 72–76. BloodMNIST over 300 rounds in `results/flame_h300/` (compared at H = 250, where the
curves level off, and H = 300); MNIST at H = 150 and the backdoor on both datasets at H = 150 here.
`tri_all` and the other references come from `results/horizon300/`, `results/f2_agreement/`,
`results/clipping_h300/` and `results/backdoor/`.

```
venv/bin/python -m scripts.triage_grid jobs --datasets bloodmnist --systems cos_bin_clip flame flame_nonoise \
    --seeds 72 73 74 75 76 --rounds 300 --out results/flame_h300 > jobs.txt
venv/bin/python -m scripts.triage_grid jobs --datasets mnist --systems cos_bin_clip flame flame_nonoise \
    --seeds 72 73 74 75 76 --out results/flame >> jobs.txt
venv/bin/python -m scripts.triage_grid jobs --datasets mnist bloodmnist --systems cos_bin_clip flame flame_nonoise \
    --seeds 72 73 74 75 76 --attacks backdoor_trigger --out results/flame >> jobs.txt
bash scripts/launch_grid.sh flame jobs.txt -P 14
venv/bin/python -m scripts.triage_compare --dataset bloodmnist --raw results/horizon300/raw results/flame_h300/raw \
    results/clipping_h300/raw --candidates tri_all tri_all_clip --single cosserver_only cos_bin sr_bin \
    --refs flame flame_nonoise cos_bin_clip fltrust fedavg oracle_uniform --horizons 250 300 --out results/flame_h300
venv/bin/python -m scripts.triage_compare --dataset mnist --raw results/f2_agreement/raw results/flame/raw \
    results/clipping/raw --candidates tri_all --single cosserver_only cos_bin sr_bin \
    --refs flame flame_nonoise cos_bin_clip fltrust --horizons 150 --seeds 72 73 74 75 76 --out results/flame
venv/bin/python -m scripts.backdoor_summary --raw results/flame/raw --out results/flame
```

## `tri_all` against each system (p.p., paired by cell and seed)

| reference | BloodMNIST, H = 250 | BloodMNIST, H = 300 | MNIST, H = 150 |
|---|---:|---:|---:|
| `flame` | +6.00 (5/5) | +5.28 (5/5) | +3.28 (5/5) |
| `flame_nonoise` | +6.24 (5/5) | +5.10 (5/5) | +3.22 (5/5) |
| `cos_bin_clip` | +3.43 (5/5) | +3.34 (5/5) | +0.17 (3/5) |
| `fltrust` | +5.37 (5/5) | +3.46 (4/5) | +0.97 (5/5) |
| best single signal | +2.62 (5/5) | +2.57 (4/5) | +0.60 (3/5) |

In parentheses, seeds with `tri_all` ahead (Wilcoxon p = 0.0625 for 5/5). Per cell against `flame`
(beyond ±1 p.p.): 16 wins, 1 tie, 2 losses on BloodMNIST (worst −4.0, FLTrust-aligned at α = 0.5);
9 wins, 7 ties, 3 losses on MNIST (worst −6.9, Gaussian noise at α = 0.05). On MNIST `tri_all` and
`cos_bin_clip` are equivalent within ±1 p.p. (TOST p = 0.03). `tri_all_clip` is 1.50 p.p. above
`cos_bin_clip` on BloodMNIST at H = 250, in 5 of 5 seeds.

## Backdoor at H = 150 (α = 0.05 / 0.1 / 0.5)

| system | MNIST ASR | BloodMNIST ASR |
|---|---|---|
| `flame` | 0.99 / 0.99 / 0.01 | 0.74 / 0.97 / 0.83 |
| `flame_nonoise` | 0.99 / 0.99 / 0.01 | 0.74 / 0.97 / 0.83 |
| `cos_bin_clip` | 0.98 / 0.99 / 1.00 | 0.66 / 1.00 / 0.71 |
| `tri_all` (`results/backdoor/`) | 0.97 / 0.99 / 1.00 | 0.55 / 1.00 / 0.52 |

## Findings

- **FLAME does not match `tri_all`.** `tri_all` is 6 p.p. above it on BloodMNIST and 3.3 p.p. on
  MNIST, with every seed ahead on both, and the noise step makes no difference. The accuracy
  contribution of TRIAGE holds against the main defense that needs no root.
- **The inversion score pays off on BloodMNIST and not on MNIST.** With clipping, adding it to the
  server cosine gains 1.5 p.p. on BloodMNIST (5 of 5 seeds); without clipping, `tri_all` is 3.4 p.p.
  above `cos_bin_clip`. On MNIST the two are equivalent, consistent with the smaller complementarity
  of the signals on that dataset.
- **FLAME does not stop the trigger backdoor either**, except on MNIST at α = 0.5, where its
  clustering separates the attackers, as plain `clustering` does.
