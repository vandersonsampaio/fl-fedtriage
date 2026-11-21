# Weighting of the included clients, and Krum

`tri_all` gives the same weight to every included client. Two alternatives were run with the
same inclusion rule:

- **`tri_all_samples`**: weight proportional to the client's sample count (FedAvg weighting);
- **`tri_all_soft`**: weight equal to the mean of the two normalized signals (inversion score and
  server cosine).

Adoption criterion, set before this run: replace the uniform weight only with a gain of at least
1 p.p. over `tri_all` in at least 4 of 5 seeds on BloodMNIST at H = 250, and no mean loss above
1 p.p. on MNIST at H = 150. Krum, the fixed rule not yet run over 300 rounds, was added on
BloodMNIST.

Logistic model, 10 clients (0 and 1 Byzantine), the 19 attack cells, seeds 72–76. BloodMNIST over
300 rounds here; MNIST at H = 150 in `results/ablation/`. `tri_all` and the other references come
from `results/horizon300/` and `results/f2_agreement/`.

```
venv/bin/python -m scripts.triage_grid jobs --datasets bloodmnist --systems tri_all_samples tri_all_soft krum \
    --seeds 72 73 74 75 76 --rounds 300 --out results/ablation_h300 > jobs.txt
venv/bin/python -m scripts.triage_grid jobs --datasets mnist --systems tri_all_samples tri_all_soft \
    --seeds 72 73 74 75 76 --out results/ablation >> jobs.txt
bash scripts/launch_grid.sh ablation jobs.txt -P 14
venv/bin/python -m scripts.triage_compare --dataset bloodmnist --raw results/horizon300/raw results/ablation_h300/raw \
    --candidates tri_all --single cosserver_only cos_bin sr_bin --refs tri_all_samples tri_all_soft krum fltrust \
    --horizons 250 300 --out results/ablation_h300
venv/bin/python -m scripts.triage_compare --dataset mnist --raw results/f2_agreement/raw results/ablation/raw \
    --candidates tri_all --single cosserver_only cos_bin sr_bin --refs tri_all_samples tri_all_soft fltrust \
    --horizons 150 --seeds 72 73 74 75 76 --out results/ablation
```

## Each system against `tri_all` (p.p., paired by cell and seed)

| system | BloodMNIST, H = 250 | BloodMNIST, H = 300 | MNIST, H = 150 |
|---|---:|---:|---:|
| `tri_all_samples` | −2.22 (0/5) | −1.71 (0/5) | −1.17 (0/5) |
| `tri_all_soft` | +0.23 (3/5) | +1.42 (5/5) | +0.80 (5/5) |
| `krum` | −19.86 (0/5) | −19.27 (0/5) | — |

In parentheses, seeds ahead of `tri_all`. Per-seed means of `tri_all_soft` on BloodMNIST:
+1.20, +1.53, +0.52, −1.12, −0.96 at H = 250 and +1.41, +1.88, +0.71, +2.18, +0.92 at H = 300.

## Findings

- **The uniform weight is kept.** `tri_all_samples` loses on both datasets and every seed; its worst
  cell is sign flipping at α = 0.1 (−36.5 p.p. at H = 250), where the attackers hold many samples.
  `tri_all_soft` is level with `tri_all` on BloodMNIST at H = 250 (2 of 5 seeds at +1 p.p. or more)
  and fails the criterion.
- **Soft weighting helps later and on MNIST.** At H = 300 it is 1.42 p.p. above `tri_all` on every
  seed, mostly from sign flipping at α = 0.05 (+15.0 p.p.), where it slows the late collapse by
  weighting down the included attackers; only 3 of 5 seeds reach +1 p.p. On MNIST it is 0.80 p.p.
  above on every seed. Neither result is at the horizon fixed by the criterion.
- **Krum is the weakest fixed rule over 300 rounds**, about 20 p.p. below `tri_all` on every seed and
  below it in every cell (worst −35.5 p.p., krum collusion at α = 0.05).
