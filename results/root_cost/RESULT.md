# Cost of the root dataset

The server cosine used by `cos_bin`, `fltrust` and `tri_all` is computed against an update the
server trains on its root dataset. The default root is 100 samples from the server validation set.
Here each system is run again with a worse root and compared with itself on the default root:

| tag | root |
|---|---|
| `r20` | 20 samples from the server validation set |
| `r10` | 10 samples from the server validation set |
| `c2-r100` | 100 samples from the training data of client 2 (honest, non-iid) |
| `r100-n0.2` | 100 samples from the server validation set, 20% of the labels replaced by another class |

Logistic model, 10 clients (0 and 1 Byzantine), the 19 attack cells, H = 150, seeds 72–76. The
default-root runs come from `results/bloodmnist_reference/` and `results/f2_agreement/` (MNIST
`fltrust` from `results/c0b_reference/`).

```
venv/bin/python -m scripts.root_compare --dataset bloodmnist \
    --raw results/bloodmnist_reference/raw results/root_cost/raw --out results/root_cost
venv/bin/python -m scripts.root_compare --dataset mnist \
    --raw results/f2_agreement/raw results/root_cost/raw --out results/root_cost
```

## Loss against the default root, same system (p.p.)

BloodMNIST:

| system (default root) | `r20` | `r100-n0.2` | `r10` | `c2-r100` |
|---|---:|---:|---:|---:|
| `tri_all` (0.634) | −0.95 | −1.43 | −4.53 | −6.54 |
| `cos_bin` (0.600) | −6.51 | −3.85 | −7.96 | −18.07 |
| `fltrust` (0.580) | −1.99 | −4.79 | −7.04 | −16.62 |

MNIST:

| system (default root) | `r20` | `r100-n0.2` | `r10` | `c2-r100` |
|---|---:|---:|---:|---:|
| `tri_all` (0.883) | −0.90 | −1.21 | −1.67 | −7.37 |
| `cos_bin` (0.877) | −3.83 | −3.03 | −7.16 | −36.86 |
| `fltrust` (0.873) | −8.78 | −5.44 | −15.98 | −44.01 |

Every loss on MNIST except `tri_all` with `r20` (4 of 5 seeds, p = 0.125), and all `c2-r100`
losses, have every seed behind the default root (Wilcoxon p = 0.0625).

## `tri_all` against the other systems on the same root (p.p.)

BloodMNIST:

| root | `tri_all` | vs `cos_bin` | vs `fltrust` | worst `tri_all` cell vs its default root |
|---|---:|---:|---:|---|
| default | 0.634 | +3.41 | +5.42 | — |
| `r20` | 0.624 | +8.96 | +6.46 | −6.1 (trim attack, α = 0.05) |
| `r100-n0.2` | 0.619 | +5.82 | +8.79 | −5.2 (Gaussian noise, α = 0.5) |
| `r10` | 0.588 | +6.84 | +7.94 | −23.5 (trim attack, α = 0.05) |
| `c2-r100` | 0.568 | +14.93 | +15.50 | −46.6 (trim attack, α = 0.05) |

MNIST:

| root | `tri_all` | vs `cos_bin` | vs `fltrust` | worst `tri_all` cell vs its default root |
|---|---:|---:|---:|---|
| default | 0.883 | +0.60 | +0.97 | — |
| `r20` | 0.874 | +3.54 | +8.86 | −2.9 (trim attack, α = 0.5) |
| `r100-n0.2` | 0.871 | +2.42 | +5.20 | −9.6 (Gaussian noise, α = 0.5) |
| `r10` | 0.866 | +6.09 | +15.28 | −8.0 (label flipping, α = 0.1) |
| `c2-r100` | 0.809 | +30.09 | +37.62 | −48.5 (trim attack, α = 0.1) |

Every seed favors `tri_all` in every row with a modified root on both datasets (p = 0.0625); on
the default root this holds on BloodMNIST, while on MNIST the gain over `cos_bin` is 3 of 5 seeds
(p = 0.31). The per-system tables, including the worst
cell of each system, are in `root_compare_<dataset>.txt` and the CSVs next to it.

## Findings

- **`tri_all` is robust to a small or noisy root.** With 20 samples or with 20% wrong labels it
  loses about 1 p.p. on both datasets, while `cos_bin` and `fltrust` lose 2–9 p.p., so its
  advantage over them grows from 1–5 to 2–9 p.p. This is consistent with the inversion score,
  which does not use the root, covering for the noisier cosine.
- **10 samples is unstable on BloodMNIST.** The `tri_all` loss on trim attack at α = 0.05 is
  bimodal: about −55 p.p. on seeds 73 and 75 and within ±7 p.p. on the other three. On MNIST,
  `tri_all` loses only 1.7 p.p. with 10 samples (worst cell −8.0, label flipping at α = 0.1,
  uniform across seeds), against 7 and 16 p.p. for `cos_bin` and `fltrust`.
- **A root drawn from a single client removes most of the gain.** With `c2-r100`, `tri_all` falls
  to 0.568 on BloodMNIST, level with the inversion score alone (`sr_bin`, 0.562), which does not
  use the root, and to 0.809 on MNIST (−7.4 p.p.). The MNIST loss is concentrated on trim attack,
  bimodal across seeds (−72 to −78 p.p. on three seeds at α = 0.1, −5 and −13 on the others).
  `cos_bin` and `fltrust` fall below `fedavg` without any defense (0.554) on BloodMNIST and lose
  37–44 p.p. on MNIST, so `tri_all` stays 15 p.p. (BloodMNIST) and 30–38 p.p. (MNIST) ahead of
  them on that root.
- **The pattern of `cos_bin` and `fltrust` is the same on both datasets and stronger on MNIST**,
  where `fltrust` loses 16 p.p. with 10 samples.
- The gain of `tri_all` therefore requires a root that is representative of the population of
  clients, of about 20 samples or more; one drawn from a single non-iid client is not enough,
  although `tri_all` degrades far less than the systems that rely on the cosine alone.
