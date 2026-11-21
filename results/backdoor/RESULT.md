# Trigger backdoor

The `low_mag_backdoor` cell of the default grid only shifts one coordinate of the parameter vector
and has no trigger, so it has no attack success rate to measure. This experiment uses a trigger
backdoor in the style of BadNets (`backdoor_trigger` in `src/classification/attack_simulator.py`):

- every round, the Byzantine clients 0 and 1 apply a 4×4 checkerboard of 0 and 1 to the
  bottom-right corner (one pixel from the border, every channel) of half of their training images
  and relabel them as class 0; the other half stays clean, and the update is not scaled;
- the attack success rate (ASR) is the fraction of the test images of all clients whose label is
  not 0 that the global model assigns to class 0 once the trigger is applied, recorded every round.

Logistic model, 10 clients, α ∈ {0.5, 0.1, 0.05}, H = 150, seeds 72–76, 13 systems on MNIST and
BloodMNIST (390 runs).

```
venv/bin/python -m scripts.triage_grid jobs --datasets mnist bloodmnist \
    --systems tri_all tri_all_tm tri_all_cl sr_bin cos_bin cosserver_only fltrust fedavg \
              trimmed_mean median clustering krum oracle_uniform \
    --seeds 72 73 74 75 76 --attacks backdoor_trigger --out results/backdoor > jobs.txt
bash scripts/launch_grid.sh backdoor jobs.txt -P 14
venv/bin/python -m scripts.backdoor_summary --raw results/backdoor/raw --out results/backdoor
```

## ASR at H = 150 (mean over 5 seeds; α = 0.05 / 0.1 / 0.5)

| system | MNIST | BloodMNIST |
|---|---|---|
| `oracle_uniform` (honest clients only) | 0.01 / 0.00 / 0.01 | 0.01 / 0.01 / 0.01 |
| `fedavg` | 0.96 / 0.99 / 1.00 | 0.63 / 1.00 / 0.80 |
| `median` | **0.01 / 0.01 / 0.01** | 0.35 / 0.71 / 0.38 |
| `trimmed_mean` | **0.02 / 0.01 / 0.01** | 0.59 / 0.72 / 0.41 |
| `krum` | 1.00 / 0.05 / 0.01 | 0.01 / 0.04 / 0.04 |
| `clustering` | 0.98 / 1.00 / **0.01** | 0.76 / **0.01** / 0.88 |
| `fltrust` | 0.51 / 0.86 / 0.66 | 0.68 / 1.00 / 0.34 |
| `cos_bin` | 0.98 / 0.99 / 1.00 | 0.44 / 0.32 / **0.02** |
| `cosserver_only` | 0.98 / 1.00 / 1.00 | 0.21 / 0.36 / **0.04** |
| `sr_bin` | **0.01 / 0.00** / 0.98 | 0.31 / **0.03 / 0.03** |
| `tri_all` | 0.97 / 0.99 / 1.00 | 0.55 / 1.00 / 0.52 |
| `tri_all_tm` | **0.08** / 0.98 / 0.96 | 0.45 / 0.86 / 0.34 |
| `tri_all_cl` | 0.98 / 1.00 / **0.01** | 0.71 / 0.24 / 0.10 |

Clean accuracy, the ASR range over seeds and the number of seeds with ASR below 0.1 are in
`backdoor_summary.csv` and `backdoor_summary.txt`.

## Clean accuracy at H = 150 (α = 0.05 / 0.1 / 0.5)

| system | MNIST | BloodMNIST |
|---|---|---|
| `oracle_uniform` | 0.880 / 0.895 / 0.917 | 0.671 / 0.522 / 0.739 |
| `fedavg` | 0.892 / 0.891 / 0.920 | 0.657 / 0.594 / 0.731 |
| `median` | 0.788 / 0.856 / 0.910 | 0.421 / 0.457 / 0.656 |
| `trimmed_mean` | 0.850 / 0.877 / 0.915 | 0.493 / 0.475 / 0.665 |
| `krum` | 0.346 / 0.463 / 0.883 | 0.359 / 0.368 / 0.535 |
| `sr_bin` | 0.877 / 0.866 / 0.910 | 0.668 / 0.522 / 0.734 |
| `tri_all` | 0.902 / 0.907 / 0.918 | 0.672 / 0.546 / 0.720 |

## Findings

- **The backdoor is effective and stealthy.** Without defense the ASR reaches 0.96–1.00 on MNIST
  and 0.63–1.00 on BloodMNIST, with clean accuracy at the level of the honest-only oracle. Clean
  accuracy alone does not reveal it.
- **`tri_all` does not stop it on either dataset.** Its ASR stays close to `fedavg`. On BloodMNIST
  it is above each of its two signals alone at every α: `tri_all` excludes a client only when both
  signals flag it, and each signal flags the attackers in different settings.
- **Each signal detects the backdoor only in part.** The inversion score (`sr_bin`) blocks it at
  α = 0.1 and 0.05 on MNIST and at α = 0.1 and 0.5 on BloodMNIST, without loss of clean accuracy.
  The server cosine blocks it only on BloodMNIST at α = 0.5. On MNIST the cosine never separates the
  attackers, whose updates are half clean and change only the weights of the trigger pixels.
- **No system blocks it everywhere without losing clean accuracy.** `median` and `trimmed_mean`
  block it at every α on MNIST but not on BloodMNIST; `krum` blocks it on BloodMNIST at the cost of
  15–31 p.p. of clean accuracy against the oracle.
- **The two-level variants help only in isolated settings.** `tri_all_cl` follows `clustering`
  (blocked on MNIST at α = 0.5 only); `tri_all_tm` blocks it on MNIST at α = 0.05 only and does not
  reproduce `trimmed_mean`, and loses 5–15 p.p. of clean accuracy on BloodMNIST at α = 0.05 and 0.1.
  The trimmed mean of the second stage runs over the included clients only, so with the same trim
  ratio it trims fewer updates per coordinate; whether this is the cause is not checked, since these
  runs do not log the inclusion decisions.
