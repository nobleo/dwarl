# Try the example robot

You'll need Linux x86-64 and a graphical display. [Pixi](https://pixi.sh) is a
package and environment manager: it installs DWARL's tools and libraries in the
checkout, so you don't need to install Python or ROS system-wide.

If you don't have Pixi, run its official Linux installer:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Open a new terminal so `pixi` is on your `PATH`, then run:

```bash
git clone https://github.com/nobleo/dwarl.git
cd dwarl
pixi run demo
```

The demo sets up the TurtleBot3 example and opens the MuJoCo viewer so you can
watch training. To change the map counts, parallel environments or simulation
speed, edit the `demo` task in `pixi.toml` before launching it.

Press **Ctrl+C** when you’re done watching. Run the printed test command to watch
the latest completed checkpoint navigate. If no checkpoint has been saved yet,
follow the printed resume command instead. After testing, open the report with
the `dwarl report` command printed for that run.

## When you're done

From the `dwarl` checkout, run `pixi clean` to remove its installed environments.
This keeps your code and training runs. The next `pixi run` recreates the environment.
To reclaim more disk space, `pixi clean cache` also clears Pixi's shared download
cache; other Pixi projects may need to download packages again.

Prefer a container? The [Docker demo](https://github.com/nobleo/dwarl/tree/main/docker)
can run the built-in example. Docker support is best-effort; native Pixi is the
supported path for development and robot integration.
