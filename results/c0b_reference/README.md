# C0b reference tables (MNIST, H = 150)

Copied from fl-gradf, `results/c0b_espaco_h150/` (last changed in commit `c8a3284`, tag `experiments-2.0.0`):

| file | sha256 |
|---|---|
| `grade_raw.csv` | `14f9079446cf4949deef567aa1b2e87dffce6afc206960624da6a7e76098de93` |
| `tetos_raw.csv` | `a1f4a154464507d4a979e03d9ea9ce9db5cdec42ac33be7342f332240152774a` |

- `grade_raw.csv`: accuracy at H ∈ {15, 50, 150} of `sr_only`, `sr_bin`, `sr_b025`, `cosserver_only`, `fltrust`, `trimmed_mean`, `clustering` and `median` on the 19 cells (α ∈ {0.5, 0.1, 0.05} × 7 attacks, without `fltrust_aligned` at α ∈ {0.1, 0.05}), seeds 72–81, logistic model, clients 0 and 1 Byzantine, root of 100 samples.
- `tetos_raw.csv`: `teto_fedavg10` (FedAvg with 10 clients, no attack) and `teto_oraculo_fedavg8` (FedAvg over the 8 honest clients) per α and seed.

These runs are deterministic: `scripts/triage_grid.py` reproduces them with |Δ| = 0 (`results/grid_parity_c0b/`: seed 72, 19 cells, H = 15, for `fltrust`, `trimmed_mean`, `clustering`, `median`, `cosserver_only` and both ceilings; `results/triage_parity/` for the inversion variants). They are used as the MNIST references for seeds 72–81.
