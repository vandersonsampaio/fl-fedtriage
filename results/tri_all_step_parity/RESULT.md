# Result — `tri_all` and the cosine signals under FLTrust step parity, root 100

**Date:** 2026-11-19. **Status: exploratory and descriptive, on development seeds 72–76; it confirms nothing.** Plan and reading rule: `PLAN.md` (hashes in `PLAN.sha256`, written before the runs). 570/570 jobs, no failure (`dispatch_vm2.log`, `dispatch_vm3.log`; per-job logs in `logs_vm2/` and `logs_vm3/`, numbered per VM), run on the two Azure VMs from 2026-11-18 21:32 to 2026-11-19 14:41 BRT (the VM logs are in UTC). Native references: the B1 runs (`results/inclusion_mechanism/raw/`, same VMs). Analysis: `scripts/analyze_step_parity.py`, output in `analysis/`.

## Reading rule

| Side | Contrast | Value | Verdict |
|---|---|---|---|
| H1 (BloodMNIST, H = 250) | `tri_all@par` − best single signal under parity (`cos_bin@par`) | +4.31 p.p., CI95 [+2.64; +5.98], 5/5 seeds | **not at risk**: CI95 above 0 and above half of the native contrast (+2.76) |
| H4 (MNIST, H = 150) | `tri_all@par` − `cos_bin@par` | +1.41 p.p., CI90 [+0.38; +2.45] | **not at risk**: lower bound above −1 p.p. |

By the plan, the claims of the F3 confirmation at root 100 are reported with both server step modes.

## Contrasts (p.p., mean over 19 cells per seed, CI95 with t over 5 seeds)

| Dataset, H | Contrast | Mean | CI95 | Seeds > 0 |
|---|---|---|---|---|
| BloodMNIST, 250 | `tri_all@par` − `tri_all@nat` | −0.02 | [−0.94; +0.89] | 2 |
| | `cos_bin@par` − `cos_bin@nat` | −1.57 | [−3.29; +0.15] | 0 |
| | `cosserver_only@par` − `cosserver_only@nat` | −1.56 | [−4.29; +1.17] | 1 |
| | `tri_all@par` − `cos_bin@par` | +4.31 | [+2.64; +5.98] | 5 |
| | `tri_all@par` − `cosserver_only@par` | +5.03 | [+3.31; +6.76] | 5 |
| | `tri_all@par` − `sr_bin` | +8.59 | [+6.93; +10.25] | 5 |
| | `tri_all@nat` − best native (`cos_bin`) | +2.76 | [+1.46; +4.07] | 5 |
| BloodMNIST, 300 | `tri_all@par` − best under parity (`cos_bin@par`) | +4.30 | [+3.01; +5.60] | 5 |
| | `tri_all@nat` − best native (`cosserver_only`) | +2.12 | [−1.11; +5.34] | 4 |
| MNIST, 150 | `tri_all@par` − `tri_all@nat` | +0.44 | [−0.24; +1.12] | 4 |
| | `cos_bin@par` − `cos_bin@nat` | −0.37 | [−1.16; +0.42] | 1 |
| | `tri_all@par` − best under parity (`cosserver_only@par`) | +0.93 | [−0.06; +1.93] | 5 |
| | `tri_all@nat` − best native (`cos_bin`) | +0.60 | [−0.74; +1.95] | 3 |

## Descriptive

- **By α (BloodMNIST, H = 250, against `cos_bin@par`):** +9.30 at α = 0.05, +1.34 at α = 0.1, +2.57 at α = 0.5. The advantage is concentrated where heterogeneity is strongest.
- **Against FLTrust under parity** (`results/fltrust_parity/raw`, run on the PC, so cross-machine and only a rough reference): +11.33 p.p. on BloodMNIST at H = 250 and +2.21 on MNIST at H = 150.
- **Mechanism of the change:** parity barely moves `tri_all` but lowers both cosine-only systems on BloodMNIST, so the gap widens because the references fall, not because `tri_all` improves. On MNIST nothing moves beyond noise.
- **Not computed in this pass** (planned as descriptive): inclusion rates under parity and the late-collapse count up to 300 rounds. The logs are in `raw/` for both.
- **Relation to the P2 framework's Front 1 check** (fl-gradf), where `cos_server` improved under parity and overtook S_R: that was a different setting (P2's framework, soft weighting, its own cells). Here, with the TRIAGE grid, the cosine-only systems lose slightly under parity on BloodMNIST.

## Consequence

- For F4: adopting step parity does not put `tri_all`'s claims at risk at root 100. The root-20 grid (`results/tri_all_root20/`, running) gives the other half.
- Nothing in `tri_all`'s configuration changes.
