# Try the example robot

On Linux x86-64 with a graphical display, install [pixi](https://pixi.sh) and run:

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
