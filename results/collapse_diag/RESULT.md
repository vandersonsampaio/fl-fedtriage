# Late collapse of `tri_all` on sign flipping (BloodMNIST, α = 0.05)

At H = 300 (`results/horizon300/`), `tri_all` on sign flipping at α = 0.05 stays near 0.65 until
round 200 and falls to about 0.49 at round 300 on all five seeds, while the honest-only oracle
reaches 0.70. This run repeats that cell with the per-client log to see who is included.

BloodMNIST, logistic model, 10 clients (0 and 1 Byzantine), sign flipping, α = 0.05, 300 rounds,
seeds 72–76. Accuracy is identical to the runs of `results/horizon300/`.

```
venv/bin/python -m scripts.triage_grid jobs --datasets bloodmnist --systems tri_all --seeds 72 73 74 75 76 \
    --attacks sign_flipping --rounds 300 --log --out results/collapse_diag | grep "alpha 0.05 " > jobs.txt
bash scripts/launch_grid.sh collapse_diag jobs.txt -P 5
venv/bin/python -m scripts.inclusion_diag --logs "results/collapse_diag/raw/bloodmnist/tri_all/*.log.csv.gz" \
    --out results/collapse_diag/inclusion.txt
```

The same cell was then run with `tri_all_clip` (`--systems tri_all_clip`; summary in
`inclusion_clip.txt`), to check whether bounding the norm of the included updates removes the
collapse.

## Inclusion and flags by window of rounds (fraction of client-rounds)

| rounds | Byzantine included | inversion flags Byzantine | cosine flags Byzantine | honest included | Byzantine / honest update norm |
|---|---:|---:|---:|---:|---:|
| 1–50 | 0.77 | 0.24 | 0.96 | 0.90 | 1.0 |
| 51–100 | 0.96 | 0.04 | 1.00 | 0.96 | 3.2 |
| 101–150 | 1.00 | 0.00 | 1.00 | 1.00 | 3.8 |
| 151–200 | 1.00 | 0.00 | 1.00 | 1.00 | 3.9 |
| 201–250 | 0.95 | 0.05 | 1.00 | 1.00 | 4.6 |
| 251–300 | 0.97 | 0.03 | 1.00 | 1.00 | 7.1 |

## Findings

- **The Byzantine clients are included from about round 50 on.** The server cosine flags them in
  almost every round, but the inversion score almost never does: their score is above that of the
  honest clients (0.93–0.98 against 0.73–0.84). `tri_all` excludes a client only when both signals
  flag it.
- **Their update norm grows without bound.** Each attacker has highly skewed data; its flipped update
  moves the model away from that data, which enlarges its next honest update and hence its next
  flipped one. The ratio to the honest norm goes from 1.0 to 7.1.
- **The uniform mean does not bound the scale of the included updates.** While the ratio is below
  about 4 the two attackers are outweighed; beyond it they dominate the aggregate, and accuracy
  falls after round 200. The wrong inclusion starts early; only its damage is late, so H = 150 hides
  it.

## With norm clipping (`tri_all_clip`)

| system | H = 150 | H = 250 | H = 300 |
|---|---:|---:|---:|
| `oracle_uniform` | 0.671 | 0.697 | 0.702 |
| `tri_all` | 0.625 | 0.632 | 0.491 |
| `tri_all_clip` | 0.558 | 0.643 | 0.666 |
| `cosserver_only` | 0.594 | 0.569 | 0.630 |

- **Clipping removes the collapse.** At round 300 `tri_all_clip` is 17.5 p.p. above `tri_all` on
  every seed (0.65–0.68 against about 0.49).
- **It does not improve the exclusion.** The Byzantine clients are still included in 65–90% of the
  rounds, but the norm ratio stays near 2 instead of growing to 7, which breaks the feedback loop.
- **It converges more slowly.** At round 150 it is 6.7 p.p. below `tri_all` in this cell.

