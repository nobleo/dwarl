"""Single RSL-RL integration boundary with ordered checkpoint and logging callbacks."""


class RunnerAdapter:
    def __init__(self, runner):
        self.runner = runner
        self.checkpoint_state = []
        self.after_save = []
        self.after_log = []
        self.writer_ready = []
        save = runner.save

        def save_checkpoint(path, infos=None):
            infos = dict(infos or {})
            for capture in self.checkpoint_state:
                infos.update(capture())
            from pathlib import Path
            from uuid import uuid4

            destination = Path(path)
            destination = destination.with_name(f"{destination.stem}_{uuid4().hex[:12]}{destination.suffix}")
            temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
            try:
                save(str(temporary), infos)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            for callback in self.after_save:
                callback(str(destination))

        runner.save = save_checkpoint
        logger = getattr(runner, "logger", None)
        if logger is not None:
            log = logger.log

            def log_iteration(*args, **kwargs):
                log(*args, **kwargs)
                it = kwargs["it"] if "it" in kwargs else args[0] if args else runner.current_learning_iteration
                for callback in self.after_log:
                    callback(int(it))

            logger.log = log_iteration
            initialize = getattr(logger, "init_logging_writer", None)
            if initialize is not None:
                def init_writer():
                    initialize()
                    if logger.writer is not None:
                        for callback in self.writer_ready:
                            callback(logger.writer)

                logger.init_logging_writer = init_writer

    @classmethod
    def for_runner(cls, runner):
        adapter = getattr(runner, "_dwarl_adapter", None)
        if adapter is None:
            adapter = cls(runner)
            runner._dwarl_adapter = adapter
        return adapter
