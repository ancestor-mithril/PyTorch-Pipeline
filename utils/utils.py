import random

import numpy as np
import torch


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
    except:
        return x


def try_trace(x):
    try:
        return torch.jit.trace(x)
    except:
        return x


def try_optimize(x, optimization: str = 'script'):
    if optimization is 'script':
        return try_script(x)
    elif optimization is 'trace':
        return try_trace(x)
    elif optimization is 'compile':
        return torch.compile(x)
    raise NotImplementedError(f'Optimization {optimization} not implemented')
