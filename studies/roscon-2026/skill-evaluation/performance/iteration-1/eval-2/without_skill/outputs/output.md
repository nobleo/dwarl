# Decision

Do not claim an improvement: 40.6% on workload A and 81.3% on workload B are not comparable. The difference may reflect workload difficulty rather than the reward change. Retain the incumbent; do not promote the trial yet.

Stop the current training sweep. Two comparable trials without useful gain exhaust a patience-of-two budget. Training indefinitely toward 95% is not a bounded decision rule; 95% can remain an aspiration, not a reason to override stopping.

## Bounded next check

Authorize evaluation only: zero additional training iterations, one incumbent–candidate pair on a frozen validation workload, followed by one repeat pair only if the candidate passes. Maximum: four evaluations, each with the same at-most-32 balanced cases. Reserve held-out tests until selection is frozen.

Before running, inspect existing summary.json, episodes.json, trace.json, compute.json, parameter snapshots and launch logs. Freeze checkpoint hashes, source revision, robot/sensor/command contract, validation case IDs and seeds, stack parameters, timeout and scoring rules. Use a clean, recorded source state; preserve existing runs and policies.

Hypothesis: the reward change improves navigation on the same workload. Acceptance: strictly more successful cases, no increase in collision rate, no degradation of the predeclared clearance threshold or recovery outcomes, and the same result direction on the repeat. Record counts and uncertainty; a small repeated gain is not proof of broad generalization. Keep scoring fixed. Reject or mark inconclusive if criteria fail; do not restart training automatically. Attribute differences to the reward only if other recipe differences are controlled.

## Commands for the next check — not executed

The exact test flags are unverified; inspect help and substitute the documented options below rather than assuming flag names.

```bash
git status --short
git rev-parse HEAD
sha256sum /path/to/incumbent/checkpoint /path/to/candidate/checkpoint
uv run dwarl test --help
```

Run these templates sequentially, using separate new output directories and identical frozen validation settings. Evaluate DWARL only; disable optional compute profiling and intermediate training checks. Do not run training concurrently or rebuild a library used by a live worker.

```bash
# Replace every placeholder with options confirmed by --help.
uv run dwarl test <incumbent-checkpoint-options> <frozen-validation-options> <baseline-output-options>
uv run dwarl test <candidate-checkpoint-options> <identical-frozen-validation-options> <candidate-output-options>

uv run python scripts/audit_comparison.py /path/to/baseline-test-directory
uv run python scripts/audit_comparison.py /path/to/candidate-test-directory

# Only if the first pair meets acceptance: repeat both with the same protocol,
# separate output directories, and the same paired repeat seeds/case IDs.
```

Audit the repeat outputs too. Compare navigation completion, collision and clearance outcomes, timeout/recovery counts and intervention activity—not process exit status alone. Report simulation elapsed time separately from wall-clock time. If validation passes twice, freeze the selected configuration and run the reserved held-out check; require Gazebo verification before promotion if these checks used another evaluator. Any further training needs a new hypothesis and an explicitly approved finite budget.
