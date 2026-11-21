# Result — BloodMNIST ceilings at H = 150

**Date:** 2026-10-11.
**Goal:** the no-attack ceilings of BloodMNIST in the in-house framework at the horizon used for MNIST (H = 150), for the logistic model and the CNN.

## Setup

Same as `results/bloodmnist_sanity/` (FedAvg, 10 clients, no attack, seed 42, CPU), with 150 rounds. CNN splits ran as separate processes in parallel (`--splits`). Script: `scripts/bloodmnist_sanity.py --rounds 150 --out results/bloodmnist_ceiling`.

## Result (global accuracy)

| model | split | round 50 | round 150 | mean of rounds 141–150 |
|---|---|---|---|---|
| logistic | iid | 0.774 | 0.787 | **0.795** |
| logistic | α = 0.5 | 0.733 | 0.757 | **0.743** |
| logistic | α = 0.1 | 0.535 | 0.604 | **0.603** |
| logistic | α = 0.05 | 0.650 | 0.656 | **0.653** |
| logistic | centralized | 0.620 | 0.776 | 0.741 (std 0.096) |
| CNN | iid | 0.832 | 0.876 | **0.877** |
| CNN | α = 0.5 | 0.793 | 0.858 | **0.858** |
| CNN | α = 0.1 | 0.660 | 0.750 | **0.746** |
| CNN | α = 0.05 | 0.688 | 0.744 | **0.742** |
| CNN | centralized | 0.888 | 0.892 | **0.891** |

## Reading

- **Ceilings for the attack grids** (FedAvg-10 without attack, mean of the last 10 rounds): logistic 0.60–0.80, CNN 0.74–0.88 depending on the split.
- The CNN has plateaued (std ≤ 0.002 over the last 10 rounds) and is 8–14 p.p. above the logistic model on every split. Its federated iid run ends 1.4 p.p. below the centralized one.
- The logistic model is close to a plateau but noisy on the non-IID splits (std up to 0.024 at α = 0.5). The centralized logistic model keeps oscillating (std 0.096) and is not a valid ceiling.
- The official AdaAggRL code reached 0.778 on BloodMNIST without attack (fl-gradf, B3.0). The in-house CNN ceiling is in the same range on iid and α = 0.5.
- **Determinism across processes:** rounds 1–50 are identical (|Δ| = 0, 500 rows) to `results/bloodmnist_sanity/`, which ran in other processes with a different degree of parallelism, for both models.

## Limits

- One seed and one partition per α (α = 0.1 below α = 0.05 is a property of these two draws).
- CPU only; bitwise reproducibility on GPU was not checked.
