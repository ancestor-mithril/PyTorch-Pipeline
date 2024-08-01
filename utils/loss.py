from torch import nn


def init_criterion(args):
    criterion = args.criterion if hasattr(args, 'criterion') else 'CrossEntropyLoss'
    reduction = args.reduction if hasattr(args, 'reduction') else 'mean'

    if criterion == 'crossentropy':
        return nn.CrossEntropyLoss(reduction=reduction)

    raise NotImplementedError(f'Criterion {args.criterion} not implemented')
