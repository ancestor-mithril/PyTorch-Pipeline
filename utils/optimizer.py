import torch
from torch import optim
from torch.optim import Optimizer


def init_optimizer(args, parameters) -> Optimizer:
    # TODO: Add fused
    momentum = 0.9 if not hasattr(args, 'momentum') else args.momentum
    weight_decay = 5e-4 if not hasattr(args, 'weight_decay') else args.weight_decay

    kwargs = {
        'lr': args.lr,
        'momentum': momentum,
        'weight_decay': weight_decay,
    }
    if torch.torch_version.TorchVersion(torch.__version__) >= '2.3.0':
        if args.device != 'cpu':
            kwargs['fused'] = True
    else:
        import warnings
        warnings.warn("Upgrade torch to support fused optimizers")

    return optim.SGD(parameters, **kwargs)
