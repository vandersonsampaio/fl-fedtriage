# Late collapse of `tri_all` up to 500 rounds (block D1) — result

**Status:** exploratory and descriptive, as fixed in `PLAN.md` (written 2026-11-17, before the runs).
205 runs, all done on the two Azure VMs (2026-11-18 00:14–12:13 UTC, no failures):
`tri_all` and `cos_bin` on the 19 BloodMNIST attack cells and `ceiling_fedavg10` without attack, seeds
72–76, 500 rounds, `--log`.

```
venv/bin/python scripts/analyze_horizon500.py     # writes analysis/
```

Files: `raw/` (accuracy per round and inclusion logs), `dispatch_vm2.log`, `dispatch_vm3.log`,
`logs_vm2/`, `logs_vm3/` (job stdout; job numbers are per VM), `analysis/analysis.txt`,
`analysis/runs.csv` (one row per run), `analysis/cells.csv`, `analysis/late_collapse_cells.csv`,
`analysis/window_rates.csv`.

## Determinism

Rounds 1–300 of all 190 `tri_all` and `cos_bin` runs reproduce the B1 runs
(`results/inclusion_mechanism/`, also run on the VMs) exactly: max |Δ accuracy| = 0 over 57,000
rounds, and the inclusion logs are identical in all 190 runs.

## Accuracy by horizon (mean over 19 cells × 5 seeds)

| system | H = 250 | H = 300 | H = 400 | H = 500 |
|---|---:|---:|---:|---:|
| `ceiling_fedavg10` (no attack) | 0.688 | 0.683 | 0.693 | 0.701 |
| `tri_all` | 0.649 | 0.646 | 0.643 | 0.652 |
| `cos_bin` | 0.622 | 0.608 | 0.636 | 0.629 |
| `tri_all` − `cos_bin` (p.p.) | +2.76 | +3.83 | +0.63 | +2.33 |
| same, without `sign_flipping` α 0.05 | +2.71 | +4.58 | +1.82 | +3.56 |

Per seed, `tri_all` − `cos_bin` at H = 500: +2.87, +1.00, +6.55, +0.98, +0.26 p.p. (5/5 positive).
At H = 400 it is 4/5 (−0.64 on seed 73). Accuracy is not pooled with `results/horizon300` (run on
the PC), as the plan fixes.

## Collapses (definition of `PLAN.md`: MA10 ≥ 5 p.p. below its running maximum, r ≥ 60)

- `ceiling_fedavg10` never collapses (0/15), so no drift without attack reaches the threshold.
- `cos_bin` meets the definition in 90 of 95 runs, almost always early (median onset 60–135 in 18
  of 19 cells; 5 late onsets in total). Its curve oscillates — it excludes 44% of honest
  clients per round over the grid — and a 5 p.p. dip of MA10 is enough. Its mean
  (final MA10 − peak MA10) is −0.5 to −5.5 p.p. per cell, so most of these "collapses" recover.
- `tri_all` meets it in 30 of 95 runs, 11 of them late (onset > 250).

## Reading rule

Two cells have a late collapse for `tri_all` in at least 2 of 5 seeds:

| cell | `tri_all` late | `cos_bin` any | `cos_bin` late | ceiling at same α | attributed to AND (plan) |
|---|---:|---:|---:|---:|---|
| `sign_flipping` α 0.05 | 3/5 (median onset 263) | 5/5 | 0/5 | 0/5 | **no** |
| `low_mag_backdoor` α 0.1 | 3/5 (median onset 438) | 5/5 | 0/5 | 0/5 | **no** |

**By the frozen rule, neither cell is attributed to the AND rule**, because `cos_bin` meets the
collapse definition in all 5 seeds of both cells. Counting only `cos_bin`'s late collapses (0/5 in
both), both would be attributed; that reading is not the plan's, and it is unfair to `cos_bin`,
since an early collapse precludes a late one under the definition. The rule's literal outcome is
the reported one. The other two cells seen up to H = 300 behave differently here: `label_flipping`
α 0.1 has no late collapse for `tri_all` (2 seeds collapse, with onsets before 250).

## Descriptive reading of the two cells (not a verdict)

**`sign_flipping` α 0.05 is a real, severe collapse specific to `tri_all`.** All five seeds end at
0.39–0.42 (final − peak −24.8 to −25.9 p.p.), while `cos_bin` ends at 0.53–0.71 (−3.4 to −9.8 p.p.).
Two seeds (73, 75) count an earlier onset (85, 66) only because of a smaller dip before the fall.
The inclusion logs show the mechanism: from round 100 on, `tri_all` includes the two sign-flipping
clients (0 and 1) in 96–100% of rounds, and the norm of the included Byzantine updates grows from 1.0× the
honest mean (rounds 1–50) to 4.2× (rounds 200–250) and 6.7–7.3× after round 250, the window of the
fall. `cos_bin` includes them in about 15% of rounds, with a norm ratio of 1.2–1.5×. After round 100,
`tri_all`'s server-cosine signal flags them in every round, but the inversion signal does not in
99% of them, so the AND rule lets them in; nothing bounds their norm, and their weight in the
aggregate grows until the model breaks.

**`low_mag_backdoor` α 0.1 is a slow drift, not a break.** `tri_all` loses −2.5 to −4.0 p.p. from its
peak, about as much as `cos_bin` (−1.3 to −6.3), and ends 4.5 p.p. below it (0.580 vs 0.625). Byzantine
inclusion by `tri_all` creeps from 0.85 to 0.91 over the 500 rounds, with the norm ratio flat at
about 1.1×; `cos_bin` stays near 0.60. The "late" onsets (433–443) are the threshold being crossed
by this drift.

## Findings

- **The collapse is one cell, not a pattern.** Of the 19 cells, only `sign_flipping` α 0.05 shows a
  late break of `tri_all`, and there it is total (5/5 seeds, −25 p.p.). The plan's rule does not
  attribute it to the AND rule, because the comparison signal `cos_bin` meets the collapse definition
  everywhere. The definition, fixed before the runs, does not separate oscillation from a break.
- **The mechanism of that cell is unbounded norm growth of included Byzantine clients that
  only one of the two signals flags**,
  consistent with B1's finding that `tri_all` lets in more Byzantine clients than `cos_bin`. It
  matches `results/collapse_diag/`; norm clipping was already tested at H = 300 and not adopted
  (`results/clipping_h300/`: it costs accuracy where `tri_all` already wins).
- **`tri_all` keeps its lead over `cos_bin` up to 500 rounds** (+2.3 p.p. at H = 500, 5/5 seeds;
  +3.6 p.p. without the collapsing cell), and stays 4.9 p.p. below the no-attack ceiling.
- **The VMs are deterministic between runs** (B1 → D1, |Δ| = 0), so later VM runs can extend these
  runs and be compared round by round.
