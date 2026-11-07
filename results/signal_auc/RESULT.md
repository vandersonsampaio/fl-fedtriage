# Per-attack separability of the TRIAGE signals

Every signal is computed on the same trajectory: the `sr_bin` system (inversion score, b = 0.5),
30 rounds, seeds 72–74, the 19 attack cells, with all six signals logged for every client and
round (`scripts/triage_grid.py --log`). The AUC uses −value as the score for `is_byz`, so 0.5 means
no separation, 1 means a perfect separation, and values far below 0.5 mean the signal ranks the
Byzantine clients as *less* suspicious than the honest ones. The full tables are in `signal_auc.txt`
(mean per-round AUC, rounds ≥ 2) and `signal_auc.csv`.

```
venv/bin/python -m scripts.signal_auc --raw results/signal_auc/raw --out results/signal_auc
```

## Mean per-round AUC, averaged over α

MNIST:

| attack           | inversion | server_cosine | median_cosine | neg_norm | neg_median_distance | temporal_cosine |
|------------------|----------:|--------------:|--------------:|---------:|--------------------:|----------------:|
| fltrust_aligned  | 0.38 | **0.00** | 0.46 | **0.00** | **0.00** | 1.00 |
| gaussian_noise   | 1.00 | 0.62 | 0.25 | 1.00 | 1.00 | 1.00 |
| krum_collusion   | 0.99 | 0.87 | 0.31 | 0.18 | 0.20 | 1.00 |
| label_flipping   | 0.55 | 0.91 | 0.83 | 0.87 | 0.89 | **0.19** |
| low_mag_backdoor | 0.54 | 0.49 | 0.46 | 0.48 | 0.43 | 0.54 |
| sign_flipping    | 0.76 | 0.93 | 0.41 | 0.67 | 0.72 | **0.36** |
| trim_attack      | 0.99 | 0.91 | 0.07 | 1.00 | 1.00 | 1.00 |

BloodMNIST:

| attack           | inversion | server_cosine | median_cosine | neg_norm | neg_median_distance | temporal_cosine |
|------------------|----------:|--------------:|--------------:|---------:|--------------------:|----------------:|
| fltrust_aligned  | 0.23 | **0.11** | 0.45 | **0.00** | **0.00** | 1.00 |
| gaussian_noise   | 1.00 | 0.61 | 0.46 | 1.00 | 1.00 | 1.00 |
| krum_collusion   | 0.98 | 0.78 | 0.62 | 0.74 | 0.75 | 0.99 |
| label_flipping   | 0.64 | 0.81 | 0.73 | 0.82 | 0.84 | **0.24** |
| low_mag_backdoor | 0.45 | 0.46 | 0.49 | 0.41 | 0.43 | 0.59 |
| sign_flipping    | 0.46 | 0.96 | 0.92 | 0.60 | 0.70 | **0.41** |
| trim_attack      | 0.83 | 0.90 | 0.54 | 1.00 | 1.00 | 0.83 |

## Findings

- **No signal is good everywhere, and the failures are complementary.** `temporal_cosine` is the
  only signal that separates `fltrust_aligned` (1.00 on both datasets, where `server_cosine` and the
  norm-based signals are fully inverted), and it is near perfect on `krum_collusion`. It is inverted
  on `label_flipping` and `sign_flipping`, where `server_cosine` is the strongest signal.
- **The inversions matter for the combiner.** A signal with AUC well below 0.5 flags the honest
  clients. With `combine="any"`, pairing `temporal_cosine` with another signal can exclude honest
  clients under label or sign flipping, and pairing `server_cosine` with it can do the same under
  `fltrust_aligned`. Whether the relative-max threshold keeps this harmless has to be measured on
  accuracy, not inferred from the AUC.
- **`low_mag_backdoor` is not separated by any signal** (all between 0.41 and 0.59 averaged over α).
  Inclusion alone cannot be expected to stop it; it needs a second stage or a backdoor-specific
  metric.
- **`neg_norm` and `neg_median_distance` are redundant** (Spearman 0.97 on both datasets); one of
  them is enough. Both are perfect on `gaussian_noise` and `trim_attack` and fail on `krum_collusion`
  in MNIST (≈ 0.2), less so in BloodMNIST (≈ 0.75).
- **`median_cosine` is the weakest signal** (0.07 on `trim_attack` in MNIST) and is not carried
  forward.
- **The inversion score is strong on the noise-like attacks** (gaussian, krum, trim) and weak on the
  semantic ones (label flipping, low-magnitude backdoor), in line with the C0b results.
- **The patterns agree across the two datasets**, with BloodMNIST slightly noisier, so the
  complementarity is a property of the attacks rather than of MNIST.
