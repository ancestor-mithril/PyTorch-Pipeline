from torch import optim
from torch.optim import Optimizer


def init_optimizer(args, parameters) -> Optimizer:
    # TODO: Add fused
    momentum = 0.9 if not hasattr(args, 'momentum') else args.momentum
    weight_decay = 5e-4 if not hasattr(args, 'weight_decay') else args.weight_decay

    return optim.SGD(parameters, lr=args.lr, momentum=momentum, weight_decay=weight_decay)
