# Skill reasoning evaluation — iteration 1

Six existing outputs were graded against their unchanged `eval_metadata.json`
assertions: three cases, one output per case per condition. Each condition is
graded on the same assertions, with exact output excerpts in `grading.json`.

| Case | With skill | Without skill |
|---|---:|---:|
| Inactive limiter attribution | 3/3 | 3/3 |
| Unmatched workloads and stopping | 3/3 | 3/3 |
| New robot setup | 4/4 | 3/4 |

The setup baseline did not state that MPPI/DWB comparison requires Gazebo;
instead it proposed checking whether MuJoCo supports the requested controllers.
All other assertions pass in both conditions. Performance assertions therefore
do not discriminate the conditions in this sample.

## Scope and limitations

- Baseline agents inherited repository guidance. This is not an instruction-free
  control and does not isolate the effect of reading a skill.
- There is only one output per condition for each of three cases. These are
  reasoning checks, not robot-performance measurements or evidence of a causal
  improvement. No repeatability or statistical significance claim is justified.
- Execution transcripts were unavailable; grades concern the preserved output
  text, not independently verified execution or command validity.
- Token counts and elapsed time remain `null`, including aggregate statistics
  and deltas. Efficiency cannot be compared. Executor model identity was not
  recorded and is also `null`.
- Benchmarks average per-case pass rates equally. Sample standard deviation for
  performance is across two distinct cases, not repeated runs; the single-case
  setup standard deviation is `null`, not zero.
- The skill-assisted setup output hard-coded `spawn entity robot`. The skill was
  subsequently corrected to use `ros.gazebo_model` (default `robot`) rather than
  require that default for every robot. **The correction was not rerun.** The
  original output remains intact and the issue is recorded in its grading; none
  of the original four assertions tests this configurability requirement.
- No robot, training or simulation experiment was run during grading/packaging.

## Contents and local review

`performance/iteration-1/` and `setup/iteration-1/` each contain benchmark JSON
and Markdown, analyst notes, original case metadata, original outputs and timing
records, and evidence-backed grades. No review HTML is stored in this repository.

Local static review pages:

- [Performance review](file:///tmp/opencode/skill-evals/performance/iteration-1/review.html)
- [Setup review](file:///tmp/opencode/skill-evals/setup/iteration-1/review.html)

These local temporary paths are machine-specific. Both pages were produced with
skill-creator's `eval-viewer/generate_review.py`. Its generated HTML received a
small null-safe formatting adjustment so missing metrics and undefined standard
deviations display as unavailable; the installed skill/viewer was not modified.
Aggregation used a null-safe equivalent rather than replacing absent metrics with
zeros. The packaging script is `/tmp/opencode/package_skill_evals.py`.

Grading followed skill-creator's `agents/grader.md`, `references/schemas.md`, and
the benchmark-analysis section of `agents/analyzer.md`. Semantic assertions were
judged against complete outputs; exact cited excerpts and aggregate arithmetic
were checked programmatically. These are unblinded retrospective grades.
