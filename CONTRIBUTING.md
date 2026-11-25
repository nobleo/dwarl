# Contributing to DWARL

Open a pull request against `main` with one change per PR. For a new feature,
dependency, or breaking change, open an issue first to discuss the approach.

Use an imperative title that makes sense in the Git history: for example,
“Report missing maps before starting training”. We squash-merge using that title.

Explain why the change is needed and how you tested it. Follow the
[development checks](web/docs/contributing.md), including the component-specific
checks for ROS and dashboard changes. For training changes, compare matched
validation cases and reserve held-out cases for final evaluation.

Before merging, all required checks must pass and review conversations must
be resolved. A maintainer may merge their own PR.
