# Amendment of 2026-11-19 — queue order (PLAN.md and its hashes unchanged)

The author expects to lose access to the VMs in the coming days. Since D2's (`results/tri_all_sc/`) frozen references are VM runs, D2 was moved ahead of the rest of this grid, at the author's request.

- At 15:40 BRT (18:40 UTC in the VM logs) the root-20 dispatchers were stopped on both VMs. 28 jobs were DONE (vm2 12, vm3 16, all `native/`, BloodMNIST; 0 failures) and are copied here (`dispatch_vm*.log`, per-job logs in `logs_vm2/` and `logs_vm3/`). The 8 jobs running at the time were killed with no output and are rerun from scratch, since the dispatcher skips only jobs logged DONE.
- On each VM, `~/r20/run_r20.sh` now resumes this grid after D2's `DISPATCH_END`, restoring first the code this grid started with (`learner.py` 44d66ef8, `triage_grid.py` 315cb46d, `test_triage.py` a4a0c86a; checked by `~/r20/code_orig.sha256`), so the "no code changed" statement of the plan still holds.
- If access to the VMs ends before this grid finishes, the remaining jobs run on the PC under the same environment, with every system and both modes of a given (cell, seed) on the same machine, and the machine recorded per job. Contrast 4 (root 20 − root 100) uses VM references and becomes cross-machine for the PC jobs; it is reported as such.
