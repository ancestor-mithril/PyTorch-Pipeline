from torch import nn


def init_criterion(args):
    if not hasattr(args, 'criterion'):
        return nn.CrossEntropyLoss()

    raise NotImplementedError(f'Criterion {args.criterion} not implemented')
