# Norm clipping as a second stage

`tri_all` does not stop the trigger backdoor (`results/backdoor/`). This experiment tests one fixed
second stage against it, with an adoption criterion written before any run:

- `tri_all_clip`: `tri_all` inclusion, then every included update above the median norm of the
  included updates is scaled down to that median, and the clipped updates are averaged uniformly
  (`norm_clip` in `src/defense/aggregation_methods.py`); no noise and no tuned bound;
- `fedavg_clip`: the same clipping over all clients, as the baseline without inclusion.

Adoption required all of: (1) backdoor ASR below 0.1 in at least 4 of the 6 settings (2 datasets × 3
α) and never above `tri_all`; (2) mean loss against `tri_all` of at most 1 p.p. on the 19 cells of
each dataset, with no cell losing more than 5 p.p.; (3) mean clean-accuracy loss against `tri_all` of
at most 1 p.p. under the backdoor on each dataset; (4) `tri_all_clip` ahead of `fedavg_clip` on the
19 BloodMNIST cells in at least 4 of 5 seeds.

Logistic model, 10 clients (0 and 1 Byzantine), H = 150, seeds 72–76.

```
venv/bin/python -m scripts.triage_grid jobs --datasets mnist bloodmnist --systems tri_all_clip \
    --seeds 72 73 74 75 76 --attacks backdoor_trigger --out results/clipping > jobs.txt
venv/bin/python -m scripts.triage_grid jobs --datasets mnist bloodmnist --systems tri_all_clip \
    --seeds 72 73 74 75 76 --out results/clipping >> jobs.txt
bash scripts/launch_grid.sh clipping jobs.txt -P 14
venv/bin/python -m scripts.backdoor_summary --raw results/clipping/raw --out results/clipping
```

The run was stopped once conditions 1 and 3 had failed, so `fedavg_clip` was not run and the 19-cell
grid is incomplete (67 of the 95 MNIST runs, none on BloodMNIST).

## Backdoor at H = 150 (mean over 5 seeds; α = 0.05 / 0.1 / 0.5)

| dataset | system | ASR | clean accuracy |
|---|---|---|---|
| MNIST | `tri_all_clip` | 0.98 / 1.00 / 1.00 | 0.901 / 0.907 / 0.917 |
| MNIST | `tri_all` | 0.97 / 0.99 / 1.00 | 0.902 / 0.907 / 0.918 |
| BloodMNIST | `tri_all_clip` | 0.73 / 1.00 / 0.80 | 0.626 / 0.492 / 0.703 |
| BloodMNIST | `tri_all` | 0.55 / 1.00 / 0.52 | 0.672 / 0.546 / 0.720 |

`tri_all` values from `results/backdoor/`; per-seed values in `backdoor_summary.csv`.

## Outcome

**Not adopted.** No setting reaches ASR below 0.1 (condition 1); on BloodMNIST the ASR is above
`tri_all` at α = 0.05 and 0.5, and clean accuracy drops by 4.6 and 5.4 p.p. at α = 0.05 and 0.1
(condition 3). The trigger backdoor stays a limitation of `tri_all`, which remains the candidate.

- On MNIST the honest clients never change the weights of the trigger pixels, which are zero in
  every honest image, so a bounded push per round still accumulates over the rounds.
- On BloodMNIST clipping to the median norm also slows the honest clients, and the attackers' updates
  are not above the median often enough to be cut.
- On the 67 MNIST runs of the 19 cells that finished, `tri_all_clip` is +0.69 p.p. above `tri_all`
  on average, mostly from label flipping (+3.9 to +4.5 p.p.) and Gaussian noise at α = 0.05
  (+8.7 p.p.). These runs are incomplete and are not part of the decision.
