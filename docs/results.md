# Test results

Each test lives in `runs/<run>/tests/<test-id>/` and records the request, saved contract,
case geometry, exact start poses, checkpoint hash, runtime bundle reference, episodes
and summary. Failed episodes remain results.

Exit codes: `0` passed, `1` completed navigation workload failed, `2` infrastructure
or input failure, `130` interrupted. Infrastructure failures do not count as navigation
success or collision observations. Missing measurements are null.

Ctrl+C preserves the current case's partial trajectory with status `interrupted`;
it is excluded from completed navigation metrics. Reports are in `report/README.md`
with side-by-side trajectory images when both backends provide measurements.

MuJoCo uses physical contacts and geometry-based footprint clearance. Travelled
distance sums displacement at each control tick. Path efficiency is reference length
divided by travelled distance, capped at one, and zero for unsuccessful episodes.

Paired simulator tests must use the same case manifest. Nav2 may calculate a different
global path, so equal worlds and starts do not imply identical controller inputs.
