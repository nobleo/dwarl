"""Episode cleanup must reach Gazebo children that create their own sessions."""

import os
import subprocess
import sys
from uuid import uuid4

import psutil

from dwarl.testing.backends.gazebo import stop_owned


def test_cleanup_reaps_detached_partition_after_worker_exits():
    partition = uuid4().hex
    child = "import os,time; print(os.getpid(), flush=True); time.sleep(60)"
    worker = subprocess.Popen(
        [sys.executable, "-c",
         f"import subprocess,sys; subprocess.Popen([sys.executable, '-c', {child!r}], start_new_session=True)"],
        env=dict(os.environ, GZ_PARTITION=partition),
        stdout=subprocess.PIPE, text=True, start_new_session=True,
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    detached = None
    try:
        detached = psutil.Process(int(worker.stdout.readline()))
        worker.wait(timeout=10)
        assert os.getsid(detached.pid) != worker.pid
        stop_owned(worker, partition)
        assert not detached.is_running() or detached.status() == psutil.STATUS_ZOMBIE
        assert unrelated.poll() is None
    finally:
        if detached is not None:
            try:
                detached.kill()
            except psutil.NoSuchProcess:
                pass
        unrelated.kill()
        unrelated.wait(timeout=10)
        worker.stdout.close()
