# Result — `tri_all` and the cosine signals at root 20, native and under FLTrust step parity

**Date:** 2026-11-21. **Status: exploratory and descriptive, on development seeds 72–76; it confirms nothing.** Plan and reading rule: `PLAN.md` (hashes in `PLAN.sha256`, written before the runs); queue change: `AMENDMENT.md`. 1,140/1,140 jobs, no failure, on the two VMs (`dispatch_vm2.log`, `dispatch_vm3.log`; per-job logs in `logs_vm2/` and `logs_vm3/`, gitignored), from 2026-11-19 ~14:00 to 2026-11-21 11:41 BRT, paused from 15:40 to 23:41 on 2026-11-19 while D2 ran. As the amendment required, the grid resumed with its original code, checked by hash on each VM. Analysis: `scripts/analyze_root20.py`, output in `analysis/` (`contrasts.csv`, `verdict.txt`).

## Reading rule

| Side | Contrast | Value | Verdict |
|---|---|---|---|
| H1′ (BloodMNIST, H = 250, root 20) | `tri_all@par` − best single signal under parity (`cos_bin@par`) | +6.72 p.p., CI95 [+2.47; +10.97], 5/5 seeds | **not at risk**: CI95 above 0 and above half of the native contrast (+7.31) |
| MNIST (H = 150, root 20) | `tri_all@par` − `cos_bin@par` | +3.97 p.p., CI90 [+2.77; +5.16] | **not at risk**: lower bound above −1 p.p. |

By the plan, the root-20 claims of the F3 confirmation are reported with both server step modes.

## Contrasts (p.p., mean over 19 cells per seed, CI95 with t over 5 seeds; all at root 20)

| Dataset, H | Contrast | Mean | CI95 | Seeds > 0 |
|---|---|---|---|---|
| BloodMNIST, 250 | `tri_all@par` − `tri_all@nat` | −0.32 | [−2.50; +1.86] | 2 |
| | `cos_bin@par` − `cos_bin@nat` | +2.05 | [−2.74; +6.84] | 4 |
| | `cosserver_only@par` − `cosserver_only@nat` | +1.61 | [−3.33; +6.55] | 2 |
| | `tri_all@par` − `cos_bin@par` | +6.72 | [+2.47; +10.97] | 5 |
| | `tri_all@par` − `cosserver_only@par` | +7.74 | [+3.03; +12.45] | 5 |
| | `tri_all@par` − `sr_bin` | +6.99 | [+4.40; +9.58] | 5 |
| | `tri_all@nat` − best native (`sr_bin`) | +7.31 | [+4.80; +9.82] | 5 |
| | `tri_all@nat` − `cos_bin@nat` | +9.09 | [+4.81; +13.36] | 5 |
| BloodMNIST, 300 | `tri_all@par` − best under parity (`cos_bin@par`) | +5.04 | [+1.62; +8.46] | 5 |
| | `tri_all@nat` − best native (`sr_bin`) | +6.52 | [+5.09; +7.94] | 5 |
| MNIST, 150 | `tri_all@par` − `tri_all@nat` | +0.21 | [−0.32; +0.74] | 4 |
| | `cosserver_only@par` − `cosserver_only@nat` | −1.92 | [−4.53; +0.68] | 0 |
| | `tri_all@par` − best under parity (`sr_bin`) | +2.13 | [+0.90; +3.36] | 5 |
| | `tri_all@nat` − best native (`sr_bin`) | +1.92 | [+0.64; +3.20] | 5 |
| | `tri_all@par` − `cos_bin@par` | +3.97 | [+2.41; +5.53] | 5 |

## Root sensitivity (contrast 4: S at root 20 − S at root 100, same mode)

| Dataset, H | Mode | `tri_all` | `cos_bin` | `cosserver_only` |
|---|---|---|---|---|
| BloodMNIST, 250 | native | −1.31 [−3.49; +0.88] | −7.63 [−13.54; −1.72] | −7.47 [−12.11; −2.84] |
| | parity | −1.60 [−3.88; +0.68] | −4.01 [−7.30; −0.72] | −4.31 [−8.03; −0.58] |
| MNIST, 150 | native | −0.90 [−1.93; +0.13] | −3.83 [−6.26; −1.41] | −2.86 [−4.20; −1.53] |
| | parity | −1.13 [−2.05; −0.21] | −3.69 [−6.00; −1.37] | −4.94 [−8.06; −1.83] |

The root-100 references are B1 (native) and `results/tri_all_step_parity/` (parity), both on the same VMs.

## Descriptive

- **By α (BloodMNIST, H = 250, `tri_all@par` − `cos_bin@par`):** +8.17 at α = 0.05, +0.95 at α = 0.1, +10.42 at α = 0.5.
- **Inclusion rates** (all rounds, all runs; Byzantine included / honest excluded): BloodMNIST `tri_all` 0.562 / 0.083 native and 0.528 / 0.079 under parity, against 0.24–0.28 / 0.46–0.48 for the cosine-only systems; MNIST `tri_all` 0.535 / 0.102 and 0.524 / 0.089, against 0.31–0.36 / 0.47–0.50. The mechanism found by B1 at root 100 (fewer honest clients excluded, more Byzantine ones included) holds at root 20 in both modes.
- **Late collapse, D1 definition** (some round ≥ 60 with MA10 ≥ 5 p.p. below its running maximum; this also counts oscillation), runs out of 95: BloodMNIST up to 300 rounds, `tri_all` 25 native / 24 parity, `cos_bin` 61 / 62, `cosserver_only` 38 / 50; MNIST up to 150 rounds, `tri_all` 20 / 14, `cos_bin` 6 / 18, `cosserver_only` 5 / 7.
- **Against FLTrust at root 20** (PC runs, so cross-machine and only a rough reference): MNIST, H = 150, `tri_all@nat` − `fltrust@nat` +8.86 [+5.77; +11.94] and `tri_all@par` − `fltrust@par` +4.36 [+1.51; +7.20]. The BloodMNIST FLTrust runs at root 20 stop at 150 rounds, so there is no contrast at H = 250 or 300.

## Reading

- Step parity barely moves `tri_all` at root 20 (−0.32 p.p. on BloodMNIST, +0.21 on MNIST), as at root 100. Unlike at root 100, the cosine-only systems tend to gain under parity on BloodMNIST (+1.6 to +2.1 p.p., CIs across zero): with a 20-sample root, the server update trained with the clients' step count is less noisy than the epoch-trained one.
- `tri_all`'s margin over the best single signal stays large in both modes (+6.7 and +7.3 p.p. on BloodMNIST at H = 250, +2.1 and +1.9 on MNIST). With root 20, the best single signal is `sr_bin`, which does not use the root, in three of the four primary comparisons.
- Going from root 100 to root 20 costs `tri_all` about 1 to 2 p.p. in both modes and the cosine-only systems 3 to 8 p.p., which repeats under parity the robustness to a small root seen in `results/root_cost/`.
- For F4 (step parity adopted): neither root puts `tri_all`'s claims at risk under parity.
