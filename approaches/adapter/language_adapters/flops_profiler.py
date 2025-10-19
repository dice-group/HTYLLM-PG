from transformers import TrainerCallback
import torch.profiler

class FlopsProfilerCallback(TrainerCallback):
    def __init__(self, wait=1, warmup=1, active=3):
        self.profiler = None
        self.wait = wait
        self.warmup = warmup
        self.active = active

    def on_train_begin(self, args, state, control, **kwargs):
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=self.wait, warmup=self.warmup, active=self.active),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(args.logging_dir),
            record_shapes=True,
            with_stack=False,
            with_flops=True,
            profile_memory=True
        )
        self.profiler.__enter__()
        print(f"[FlopsProfiler] Profiling started (logging to {args.logging_dir})")

    def on_step_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            print("[FlopsProfiler] Profiling finished and written to TensorBoard.")
