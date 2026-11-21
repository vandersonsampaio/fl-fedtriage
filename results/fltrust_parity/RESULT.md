# Result — FLTrust with step parity on the development seeds

**Date:** 2026-11-03. **Status: descriptive, on development seeds; it confirms nothing.** Plan and reading rule: `ADDENDUM.md`. 380/380 jobs, no traceback, 2026-11-03 00:47 to about 06:10, 7 parallel jobs, alongside the W1 rerun of `fl-gradf`. Analysis: `scripts/analyze_fltrust_parity.py`, output in `analysis/`.

**In short:** by the frozen rule, the claims against FLTrust are **not at risk** on BloodMNIST at either root. The screen is one-sided, though: the gap widens because FLTrust falls under parity while `tri_all` stays native, so `tri_all` under parity is still unmeasured.

## Contrasts

Per seed, mean over the 19 cells, CI95 with t, 5 seeds (72–76), p.p.:

| dataset | root | H | FLTrust parity − native | `tri_all` − FLTrust native | `tri_all` − FLTrust parity | rule |
|---|---|---|---|---|---|---|
| BloodMNIST | 100 | 250 | −6.04 (−11.18; −0.90) | +5.37 (+1.45; +9.28) | +11.41 (+3.77; +19.04) | not at risk |
| BloodMNIST | 100 | 150 | −8.47 (−18.11; +1.16) | +5.42 (+0.16; +10.69) | +13.90 (+8.43; +19.36) | not at risk |
| BloodMNIST | 20 | 150 | −15.12 (−21.88; −8.35) | +6.46 (+4.72; +8.20) | +21.58 (+14.74; +28.41) | not at risk |
| MNIST | 100 | 150 | −0.79 (−2.80; +1.22) | +0.97 (+0.21; +1.73) | +1.76 (−0.36; +3.89) | outside the rule |
| MNIST | 20 | 150 | +4.71 (+1.71; +7.72) | +8.86 (+5.77; +11.94) | +4.14 (+1.00; +7.29) | outside the rule |

Check: the native BloodMNIST values at H = 150 (`tri_all` 63.37, gap +5.42) reproduce `results/horizon300/compare_bloodmnist.csv`.

## Reading

- **The prediction made before the runs failed.** The addendum expected a smaller effect on BloodMNIST, because its gap in steps is smaller. The effect is larger there.
- **Parity does not act the same way across α.** Parity − native, BloodMNIST root 100, H = 250:

| α | parity − native | `tri_all` − FLTrust parity |
|---|---|---|
| 0.05 | +7.98 | +2.79 |
| 0.10 | −17.17 | +22.08 |
| 0.50 | −8.51 | +9.65 |

- **The strongest competitor appears at low α.** At α 0.05 parity helps FLTrust and the gap is smallest. In the 18 cells outside `fltrust_aligned`, `tri_all` falls below FLTrust parity under `low_mag_backdoor` (−0.9) and `trim_attack` (−0.1) at that α. On MNIST root 100 the average gap at α 0.05 is −3.09, and across all α it is not shown to be above zero.
- **Root 20.** The large fall on BloodMNIST fits the overfitting of g₀ to the root that the addendum anticipated, which was not tested. MNIST goes the other way (+4.71). A residual root effect is not a pure content effect.

## Limits

- **`tri_all` stays native,** although it reads g₀'s direction too. `tri_all` − FLTrust parity is a screen for risk, not an estimate of the corrected contrast. If `tri_all` under parity falls as FLTrust does, the corrected gap would be smaller than this table.
- **Between-seed variation is large,** with five seeds: FLTrust parity − native at H = 150 ranges from −0.20 to −16.71 on BloodMNIST root 100.
- **Client streams shift** between parity and native (`ADDENDUM.md`), so the differences include that effect.
- **Why parity lowers FLTrust at α ≥ 0.10 on BloodMNIST** was not investigated.

## Consequence (frozen rule)

Not at risk: FLTrust's comparisons are reported with both versions. Step 2 (`tri_all` and the cosine signals under parity) is decided by the unmeasured direction question, not by this table. Seeds 77–81 were not needed by the rule.
