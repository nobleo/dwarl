# Clickable architecture slides

**Layout decision:** Keep the slide geometry instead of asking a graph algorithm to arrange the boxes. The [Vue explorer](../theory/Explorer.vue) aligns HTML buttons with SVG connectors in a common coordinate system. [Two layouts](../theory/layout.js) preserve the two input branches and the training feedback loop at phone widths. Ordinary buttons expose component names, keyboard activation and focus outlines.

**Drill-down:** DWARL changes the graph level; selecting a leaf opens a native dialog. Detail text comes from one [Markdown article](../docs/architecture.md), available as an expandable text alternative without JavaScript. Keep the original command chain in view when a detail opens. Check keyboard activation, Escape, return focus and hotspot alignment at phone and desktop widths.

**When Vue adds value:** If drill-down grows into a multi-step explorer, mount Vue only on the explorer. Vue’s [Quick Start](https://vuejs.org/guide/quick-start.html#using-vue-from-cdn) explicitly supports enhancing static HTML with a self-hosted runtime; single-file components require a build. For an actual graph canvas, Vue Flow supports [explicit node positions and disabling dragging](https://github.com/bcakmakoglu/vue-flow/blob/master/docs/src/guide/node.md), [custom node slots and core CSS](https://github.com/bcakmakoglu/vue-flow/blob/master/docs/src/guide/getting-started.md), and [`@node-click` with `{ event, node }`](https://github.com/bcakmakoglu/vue-flow/blob/master/docs/src/guide/vue-flow/events.md). A custom node can render a real link to its detail heading. These features are useful for graph exploration; fixed artwork already encodes the required positions, lettering and connector paths.

**Hosting:** [MkDocs config](../mkdocs.yml) builds `web/docs/` into `web/site/`, registers the architecture CSS/JS, and targets GitHub Pages under `/dwarl/`. [Docs CI](../../.github/workflows/docs.yml) publishes the site output; the [dashboard Vite build](../vite.config.js) writes elsewhere. Place artwork and any prebuilt explorer assets under `web/docs/`, use page-relative asset URLs that resolve on the built page, and avoid backend routes or root-relative `/assets/` links. Verify with the documented [`pixi run docs` strict build and HTTP preview](../docs/documentation.md).

### What the boxes refer to in DWARL

| Diagram connection | Source of truth |
| --- | --- |
| Path and sensor history → observation | [Lookahead sampling](../../dwarl/src/dwarl/config/recipes/commands/path.yaml), [scan lags and state terms](../../dwarl/src/dwarl/config/recipes/observations/navigation.yaml), and [native path projection and scan binning](../../ros/src/pkgs/dwarl_controller/src/policy_core.cpp) |
| Observation → actor → dynamic window | [Circular lidar encoder](../../dwarl/src/dwarl/training/models/lidar_cnn.py), [network dimensions and critic-only inputs](../../dwarl/src/dwarl/config/recipes/agent/ppo.yaml), and [velocity action](../../dwarl/src/dwarl/navigation/actions.py) |
| Simulated motion → learning signal | [Environment step](../../dwarl/src/dwarl/training/environment/nav_env.py) and [reward terms](../../dwarl/src/dwarl/config/recipes/rewards/path.yaml) |
| Exported actor → guarded robot command | [Native controller](../../ros/src/pkgs/dwarl_controller/src/controller.cpp), [launch-time command routing](../../ros/src/pkgs/dwarl_bringup/launch/navigation.launch.py), and [stop/retry tree](../../ros/src/pkgs/dwarl_bringup/behavior_trees/dwarl_safety_retry.xml) |
