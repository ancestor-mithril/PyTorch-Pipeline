import torch
from torch import nn
from torch.nn import Module


class LossIQR(Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.quantiles = torch.Tensor([0.25, 0.75])

    def forward(self, outputs, targets):
        loss = self.loss(outputs, targets)
        # IQR = Q3 - Q1. We eliminate only loss values above Q3 + 1.5 IQR = 2.5 Q3 - 1.5 Q1
        with torch.no_grad():
            q1, q3 = torch.quantile(loss, self.quantiles)
            mask = loss < 2.5 * q3 - 1.5 * q1

        return loss[mask].mean()


def init_criterion(args):
    if args.criterion == 'crossentropy':
        if args.reduction == 'iqr':
            return LossIQR(nn.CrossEntropyLoss(reduction='none'))
        return nn.CrossEntropyLoss(reduction=args.reduction)

    raise NotImplementedError(f'Criterion {args.criterion} not implemented')
