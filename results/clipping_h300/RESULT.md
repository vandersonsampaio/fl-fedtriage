# Norm clipping as scale control (BloodMNIST, 300 rounds)

`tri_all_clip` failed its first adoption criterion, written for the trigger backdoor
(`results/clipping/`). The diagnosis of the late collapse on sign flipping (`results/collapse_diag/`)
then showed a separate problem: `tri_all` does not bound the scale of the included updates, and
`tri_all_clip` removes that collapse. Clipping was therefore re-evaluated once, as scale control,
under a revised criterion written before this run. Adoption required all of, on BloodMNIST at
H = 250:

1. no cell whose mean accuracy drops by more than 5 p.p. between its best value over rounds 150–300
   and round 300;
2. a mean loss against `tri_all` of at most 1 p.p., and no cell losing more than 5 p.p. except
   cells where `tri_all` itself collapses;
3. a gain of at least 1 p.p. over the best single signal (`cosserver_only`) in at least 4 of 5 seeds;
4. `tri_all_clip` ahead of `fedavg_clip` in at least 4 of 5 seeds;
5. on MNIST at H = 150, a mean loss against `tri_all` of at most 1 p.p.

BloodMNIST, logistic model, 10 clients (0 and 1 Byzantine), the 19 attack cells, seeds 72–76, 300
rounds. `tri_all` and the other references come from `results/horizon300/`.

```
venv/bin/python -m scripts.triage_grid jobs --datasets bloodmnist --systems tri_all_clip \
    --seeds 72 73 74 75 76 --rounds 300 --out results/clipping_h300 > jobs.txt
bash scripts/launch_grid.sh clipping_scale jobs.txt -P 14
venv/bin/python -m scripts.triage_compare --dataset bloodmnist --raw results/horizon300/raw \
    results/clipping_h300/raw --candidates tri_all_clip --single cosserver_only cos_bin sr_bin \
    --refs tri_all fltrust fedavg oracle_uniform --horizons 250 300 --out results/clipping_h300
```

The run was stopped once conditions 1–3 had failed: `fedavg_clip` was not run, and six more MNIST
runs of `tri_all_clip` were added to `results/clipping/` (88 of the 95 cells).

## Outcome at H = 250

| condition | required | result |
|---|---|---|
| 1. no late collapse | drop ≤ 5 p.p. in every cell | 6.6 p.p. in one cell (`tri_all`: 16.8) |
| 2. against `tri_all` | mean loss ≤ 1 p.p.; no cell below −5 p.p. | −1.94 p.p.; worst cell −6.4 (krum collusion, α = 0.05) |
| 3. against `cosserver_only` | ≥ +1 p.p. in ≥ 4 of 5 seeds | +1.68 p.p. in 3 of 5 seeds |

**Not adopted.** Clipping slows convergence and costs accuracy on the attacks where `tri_all` already
excludes the attackers well (krum collusion and trim attack at α = 0.05, about −6 p.p. each); its
gain on sign flipping at α = 0.05 does not make up for it. `tri_all` remains the candidate, and the
missing scale control is a declared limitation. The full tables are in `compare_bloodmnist.txt`.
