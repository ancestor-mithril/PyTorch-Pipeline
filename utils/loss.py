import torch
from torch import nn
from torch.nn import Module


class LossMeanStd(Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        # DEBUG
        self.progress_tracker = []

    def forward(self, outputs, targets):
        loss = self.loss(outputs, targets)
        # loss <  1.5 std + mean
        with torch.no_grad():
            std, mean = torch.std_mean(loss)
            mask = loss < 1.5 * std + mean
            # DEBUG
            self.progress_tracker.append(mask.sum() / mask.numel())
        return loss[mask].mean()


class LossIQR(Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss
        self.register_buffer('quantiles',  torch.Tensor([0.25, 0.75]))
        self.register_buffer('weights',  torch.Tensor([-1.5, 2.5]))
        self.quantiles: torch.Tensor
        self.weights: torch.Tensor
        # DEBUG
        self.progress_tracker = []

    def forward(self, outputs, targets):
        loss = self.loss(outputs, targets)
        # IQR = Q3 - Q1. We eliminate only loss values above Q3 + 1.5 IQR = 2.5 Q3 - 1.5 Q1
        with torch.no_grad():  # TODO: Maybe use to cpu
            mask = loss < (torch.quantile(loss, self.quantiles) * self.weights).sum()
            # DEBUG
            self.progress_tracker.append(mask.sum() / mask.numel())

        return loss[mask].mean()


def init_criterion(args):
    # TODO: Trace/script/compile criterion
    if args.criterion == 'crossentropy':
        if args.reduction == 'iqr':
            return LossIQR(nn.CrossEntropyLoss(reduction='none'))
        if args.reduction == 'stdmean':
            return LossMeanStd(nn.CrossEntropyLoss(reduction='none'))

        return nn.CrossEntropyLoss(reduction=args.reduction)

    raise NotImplementedError(f'Criterion {args.criterion} not implemented')
