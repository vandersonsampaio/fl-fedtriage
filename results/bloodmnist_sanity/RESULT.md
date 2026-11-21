# Result — BloodMNIST sanity check in the in-house framework

**Date:** 2026-10-11.
**Criterion:** FedAvg with 10 clients and no attack converges on BloodMNIST, for the logistic model and the CNN; record the ceiling of each split.

## Setup

- Data: official MedMNIST v2 `bloodmnist.npz` (28×28×3, 8 classes, 11,959 train / 3,421 test), partitioned by `data/prepare_bloodmnist.py` (seed 42): `iid`, `non_iid` (Dirichlet α = 0.5), `non_iid_a0.1`, `non_iid_a0.05`; 10 clients, 597 server-validation samples per split.
- FedAvg, no attack, 50 rounds, local training as for MNIST (lr 0.01, 5 epochs, batch 32), seed 42, CPU.
- Centralized reference: one client holding the union of the 10 clients' training data (`iid` only).
- Script: `scripts/bloodmnist_sanity.py`; per-round accuracy in `logistic_seed42.csv` and `cnn_seed42.csv`.

## Result (global accuracy)

| split | logistic, round 30 | logistic, round 50 | CNN, round 30 | CNN, round 50 |
|---|---|---|---|---|
| iid | 0.756 | **0.774** | 0.794 | **0.832** |
| α = 0.5 | 0.688 | **0.733** | 0.735 | **0.793** |
| α = 0.1 | 0.506 | **0.535** | 0.593 | **0.660** |
| α = 0.05 | 0.603 | **0.650** | 0.610 | **0.688** |
| centralized (iid) | — | 0.620 (oscillates 0.41–0.81) | — | **0.888** |

## Reading

- FedAvg converges for both models on every split. Neither has plateaued at round 50: all splits still gain 2–7 p.p. between rounds 30 and 50.
- The CNN is 5–12 p.p. above the logistic model under FedAvg, and its centralized reference (0.888) bounds the federated runs from above.
- The centralized logistic model is **not** a valid ceiling at this learning rate: one model trained sequentially on all the data oscillates between 0.41 and 0.81 from round to round. Averaging 10 client models smooths that oscillation, so FedAvg ends above it.
- α = 0.1 is below α = 0.05 for both models. This is a property of the two particular partitions (one draw each), not a monotonic effect of α.

## Limits

- One seed and one partition per α.
- 50 rounds; the ceilings at the horizon used for MNIST (H = 150) are still to be measured.
