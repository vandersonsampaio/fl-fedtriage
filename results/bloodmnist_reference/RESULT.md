# BloodMNIST: references and TRIAGE configurations

Logistic model, 10 clients (0 and 1 Byzantine), the 19 attack cells (α ∈ {0.5, 0.1, 0.05}),
root of 100 samples from the server validation set, H = 150, seeds 72–76. TRIAGE configurations
use b = 0.5 (relative-max threshold), uniform weights and lam = 2 unless noted.

```
venv/bin/python -m scripts.triage_compare --dataset bloodmnist --raw results/bloodmnist_reference/raw \
    --candidates tri_all tri_any cos_temp_any cheap_any --single sr_bin cosserver_only \
    --refs cos_bin fltrust trimmed_mean clustering median fedavg oracle_uniform sr_b025 \
    --out results/bloodmnist_reference
```

## Mean accuracy at H = 150 (19 cells × 5 seeds)

| system | accuracy |
|---|---:|
| `ceiling_fedavg10` (no attack, mean over the 3 non-iid α) | 0.672 |
| `oracle_fedavg8` (honest clients only, no attack) | 0.665 |
| `oracle_uniform` (exactly the honest clients averaged) | 0.644 |
| **`tri_all`** (inversion ∧ server_cosine) | **0.634** |
| `cos_bin` (server_cosine, uniform) | 0.600 |
| `cosserver_only` (server_cosine, soft) | 0.598 |
| `fltrust` | 0.580 |
| `sr_bin` (inversion, uniform) | 0.562 |
| `cheap_any` (neg_median_distance ∨ temporal_cosine) | 0.560 |
| `tri_any` (inversion ∨ server_cosine) | 0.560 |
| `fedavg` | 0.554 |
| `cos_temp_any` (server_cosine ∨ temporal_cosine) | 0.545 |
| `clustering` | 0.536 |
| `sr_b025` (inversion, b = 0.25, soft) | 0.527 |
| `trimmed_mean` | 0.515 |
| `median` | 0.463 |

## `tri_all` against its references (paired by cell and seed, p.p.)

| reference | difference | Wilcoxon p (5 seeds) |
|---|---:|---:|
| best single signal (`cosserver_only`) | +3.57 | 0.0625 |
| per-cell oracle choice between `sr_bin` and `cosserver_only` | +1.31 | 0.0625 |
| `fltrust` | +5.42 | 0.0625 |
| `fedavg` | +7.93 | 0.0625 |
| `oracle_uniform` | −1.03 | 0.19 |

With 5 seeds, 0.0625 is the smallest two-sided Wilcoxon p-value: every seed favors `tri_all`.

## Findings

- **`tri_all` beats the per-cell choice of signal on BloodMNIST** (+1.31 p.p.) and is ahead of the
  best single signal in all 19 cells (7 beyond 2 standard errors, none behind). It is within about
  1 p.p. of averaging exactly the honest clients.
- **The two signals are more complementary here than on MNIST**: the per-cell oracle is 2.3 p.p.
  above the best single signal (1.2 p.p. on MNIST), and `sr_bin` alone collapses on sign flipping
  and on trim attack at α = 0.05, where the server cosine holds.
- **`any` combinations lose on BloodMNIST as on MNIST** (−3.8 to −5.3 p.p. against the best single
  signal), from excluding honest clients.
- **The robust aggregation rules are weak on this dataset**: `median`, `trimmed_mean` and
  `clustering` are below plain `fedavg`.

## Confirmation on seeds 77–81

`tri_all` and all the references were run again on the five held-out seeds. The output is in
`seeds_77_81/`.

```
venv/bin/python -m scripts.triage_compare --dataset bloodmnist --raw results/bloodmnist_reference/raw \
    --candidates tri_all --single sr_bin cosserver_only \
    --refs cos_bin fltrust trimmed_mean clustering median fedavg oracle_uniform \
    --seeds 77 78 79 80 81 --out results/bloodmnist_reference/seeds_77_81
```

| `tri_all` against | seeds 72–76 | seeds 77–81 | Wilcoxon p (77–81) | seeds ahead (77–81) |
|---|---:|---:|---:|---:|
| best single signal (`cosserver_only`) | +3.57 | +4.03 | 0.0625 | 5 / 5 |
| per-cell oracle | +1.31 | +1.94 | 0.19 | 4 / 5 |
| `cos_bin` | +3.41 | +2.80 | 0.125 | 4 / 5 |
| `fltrust` | +5.42 | +3.19 | 0.125 | 4 / 5 |
| `fedavg` | +7.93 | +6.95 | 0.0625 | 5 / 5 |
| `clustering` | — | +8.32 | 0.0625 | 5 / 5 |
| `trimmed_mean` | — | +11.24 | 0.0625 | 5 / 5 |
| `median` | — | +16.19 | 0.0625 | 5 / 5 |
| `oracle_uniform` | −1.03 | −1.16 | 0.125 | — |

- **The development result holds on the held-out seeds.** `tri_all` reaches 0.629 (0.634 on
  seeds 72–76) and is +4.03 p.p. above the best single signal with every seed in favor. Per cell
  it has 5 wins, 14 ties and no loss; 18 of the 19 cells are positive and the only negative one is
  −0.05 p.p. (trim attack, α = 0.5).
- **It stays above the per-cell oracle** (+1.94 p.p.), although on these seeds one of the five
  seeds is below it.
- **The largest gains are at α = 0.05 on sign flipping, Gaussian noise and Krum collusion**
  (+7 to +11 p.p.).
- **The `low_mag_backdoor` cells also show large gains** (+5 to +8 p.p.), but they measure only
  the cost of excluding honest clients: the attack changes a single coordinate of the parameter
  vector, has no trigger, and `fedavg` under it stays within 1.8 p.p. of the no-attack ceiling.
