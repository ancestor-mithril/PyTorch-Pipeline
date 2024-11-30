import random

import numpy as np
import torch
from timed_decorator.simple_timed import timed

from utils.logger import get_logger


def seed_everything(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)


def try_script(x):
    try:
        return torch.jit.script(x)
    except Exception as e:  # noqa E722
        get_logger().log_both(f"Scripting failed due to {type(e).__name__}")
        return x


def try_trace(x):
    try:
        return torch.jit.trace(x)
    except Exception as e:  # noqa E722
        get_logger().log_both(f"Tracing failed due to {type(e).__name__}")
        return x


def try_compile(x, **kwargs):
    try:
        return torch.compile(x, **kwargs)
    except Exception as e:  # noqa E722
        get_logger().log_both(f"Compiling failed due to {type(e).__name__}")
        return x


def try_optimize(x, optimization: str = "script"):
    if optimization == "script":
        return try_script(x)
    elif optimization == "trace":
        return try_trace(x)
    elif optimization == "compile":
        return try_compile(x)  # torch.compile may also raise during training
    raise NotImplementedError(f"Optimization {optimization} not implemented")


class TimeAccumulator:
    def __init__(self, fn):
        self.total = 0
        self.calls = 0
        self.fn = fn

    def __call__(self, *args, **kwargs):
        ret, elapsed = timed(return_time=True, stdout=False)(self.fn)(*args, **kwargs)
        self.total += elapsed
        self.calls += 1
        print(self.total / self.calls)
        return ret
