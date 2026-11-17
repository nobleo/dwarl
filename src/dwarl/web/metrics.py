"""Incremental, bounded TensorBoard scalar reads for native dashboard charts."""

import math
import threading
import time
from collections import OrderedDict


class Metrics:
    def __init__(self):
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def read(self, run):
        if not (run / "checkpoints").is_dir():
            return {"series": [], "max_points": 1500}
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        path = str(run / "checkpoints")
        with self.lock:
            if path not in self.cache:
                self.cache[path] = (EventAccumulator(path, size_guidance={"scalars": 1500}), 0)
            accumulator, last = self.cache.pop(path)
            if time.monotonic() - last > 3:
                accumulator.Reload()
                last = time.monotonic()
            self.cache[path] = accumulator, last
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
            series = []
            for tag in accumulator.Tags().get("scalars", []):
                points = [[event.step, event.value] for event in accumulator.Scalars(tag)
                          if math.isfinite(event.value)]
                if points:
                    series.append({"tag": tag, "points": points})
            return {"series": series, "max_points": 1500}
