# Result — `tri_all_sc`: capping only extreme norms among included clients (block D2)

**Date:** 2026-11-20. **Status: exploratory, on development seeds 72–76; it confirms nothing.** Plan and criterion: `PLAN.md` (hashes in `PLAN.sha256`, written and authorized before the runs). 190/190 jobs, no failure, on the two VMs from 2026-11-19 15:42 to 23:41 BRT (`dispatch_vm2.log`, `dispatch_vm3.log`; per-job logs in `logs_vm2/` and `logs_vm3/`, gitignored). Analysis: `scripts/analyze_tri_all_sc.py`, output in `analysis/` (`verdict.txt`, `deltas.csv`, `determinism.csv`, `caps_by_window.csv`).

**Deviation from the plan's execution section:** the plan said the D2 code would be installed only after the root-20 grid ended. At the author's request the queue was swapped (`results/tri_all_root20/AMENDMENT.md`): D2 ran first, and the root-20 grid resumed afterwards with its original code restored and checked by hash. The installed D2 code matched the hashes in the plan (commit `5183488`).

## Verdict: PASS (all three conditions)

| Condition | Rule | Result |
|---|---|---|
| 1. The collapse goes away | BloodMNIST, `sign_flipping` α 0.05, H = 500: peak-to-end drop ≤ 5 p.p. in ≥ 4/5 seeds | **5/5**: drops 1.1–2.0 p.p. (`tri_all`: 24.8–25.9) |
| 2. No meaningful cost | Mean Δ ≥ −1 p.p. at BloodMNIST H = 250 and MNIST H = 150; no cell < −5 | BloodMNIST **+0.64** (CI95 +0.17; +1.12), worst cell −0.23; MNIST **+0.48** (−0.02; +0.97), worst cell −1.34 |
| 3. No new collapse | No other BloodMNIST cell with drop > 5 p.p. in ≥ 2/5 seeds for `tri_all_sc` while < 2/5 for `tri_all` | **none**; the only other cell with drops is `trim_attack` α 0.05, where `tri_all` has 2/5 and `tri_all_sc` 0/5 |

As fixed in the plan, `tri_all` stays the configuration of the thesis whatever this outcome; the result goes to the discussion and future work.

## Δ = `tri_all_sc` − `tri_all` (p.p., mean over cells and seeds, CI95 with t over 5 seeds)

| Dataset, H | All cells | α = 0.05 | α = 0.1 | α = 0.5 | Worst cell |
|---|---|---|---|---|---|
| BloodMNIST, 250 | +0.64 [+0.17; +1.12] | +1.57 [+0.02; +3.11] | +0.13 [−0.21; +0.46] | +0.29 [+0.09; +0.49] | `low_mag_backdoor` α 0.05, −0.23 |
| BloodMNIST, 300 | +1.39 [+0.72; +2.06] | +3.66 [+1.91; +5.40] | +0.46 [−0.15; +1.07] | +0.25 [+0.05; +0.46] | `low_mag_backdoor` α 0.1, −0.46 |
| BloodMNIST, 400 | +1.70 [+0.94; +2.46] | +4.74 [+1.95; +7.53] | +0.11 [−0.89; +1.10] | +0.45 [−0.20; +1.11] | `trim_attack` α 0.05, −1.65 |
| BloodMNIST, 500 | +2.15 [+1.00; +3.30] | +5.32 [+2.85; +7.79] | +1.08 [−0.05; +2.21] | +0.35 [−0.33; +1.03] | `low_mag_backdoor` α 0.05, −0.13 |
| MNIST, 150 | +0.48 [−0.02; +0.97] | +1.07 [−0.49; +2.62] | −0.03 [−0.10; +0.04] | +0.40 [+0.29; +0.52] | `trim_attack` α 0.05, −1.34 |

The gain grows with the horizon on BloodMNIST and is concentrated at α = 0.05, where the late collapse was.

## Descriptive

- **Collapse by D1's definition** (some round ≥ 60 with MA10 ≥ 5 p.p. below its running maximum; this also counts oscillation): BloodMNIST 19/95 runs for `tri_all_sc` against 30/95 for `tri_all`; MNIST 4/95 against 10/95.
- **What gets capped.** BloodMNIST: 1.14% of included honest updates and 40.1% of included Byzantine updates; MNIST: 0.01% and 28.3%. In the target cell, the cap reaches no Byzantine update in rounds 1–50, 45% in rounds 51–100 and 100% from round 101 on; it caps 12–13% of included honest updates in every window (6% in rounds 51–100), the risk anticipated in the plan, which did not translate into a loss.
- **Determinism check:** 160 of 190 runs are identical to their reference (|Δ| = 0) in every round before the first cap; 63 of those are never capped and equal the reference over the whole run. 30 runs cap in round 1, which makes the check vacuous for them. No run differs from its reference before its first cap.

## Reading

A cap that only touches norms above 3 × the median of the included clients removes the one late collapse found up to 500 rounds without a measurable cost anywhere in the grid: in no cell does it lose more than 1.65 p.p. at any horizon. Unlike clipping to the median (`tri_all_clip`, rejected twice), it leaves almost every honest update alone. This is exploratory evidence on development seeds; making `tri_all_sc` part of a claim would need its own preregistration on fresh seeds.
