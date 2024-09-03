import os
from abc import abstractmethod
from functools import partial, lru_cache

import torch
from torch import Tensor, nn

from utils.logger import get_logger


class MeanReducer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.mean()


class NoneReducer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class LossMeanStd(nn.Module):
    def __init__(self, loss, reducer):
        super().__init__()
        self.loss = loss
        self.reducer = reducer
        # DEBUG
        self.progress_tracker = []

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        loss = self.loss(outputs, targets)
        # loss <  1.5 std + mean
        with torch.no_grad():
            std, mean = torch.std_mean(loss)
            mask = loss < 1.5 * std + mean
            # DEBUG
            self.progress_tracker.append(mask.sum() / mask.numel())
        return self.reducer(loss[mask])


class LossIQR(nn.Module):
    def __init__(self, loss, reducer):
        super().__init__()
        self.loss = loss
        self.reducer = reducer
        self.register_buffer("quantiles", torch.Tensor([0.25, 0.75]))
        self.register_buffer("weights", torch.Tensor([-1.5, 2.5]))
        self.quantiles: torch.Tensor
        self.weights: torch.Tensor
        # DEBUG
        self.progress_tracker = []

    def forward(self, outputs, targets):
        loss = self.loss(outputs, targets)
        # IQR = Q3 - Q1. We eliminate only loss values above Q3 + 1.5 IQR = 2.5 Q3 - 1.5 Q1
        with torch.no_grad():
            # TODO: Maybe use to cpu
            mask = loss < (torch.quantile(loss, self.quantiles) * self.weights).sum()
            # DEBUG
            self.progress_tracker.append(mask.sum() / mask.numel())

        return self.reducer(loss[mask])


class LossScaler(nn.Module):
    def __init__(self, loss, reducer):
        super().__init__()
        self.loss = loss
        self.reducer = reducer
        self.loss_scaling_range = float(os.getenv("loss_scaling_range", 0.25))
        get_logger().log_both(
            f"Using Loss scaling with scaling range: {self.loss_scaling_range}"
        )

    @lru_cache(maxsize=32)
    def get_weights(
        self, shape: torch.Size, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        return self._get_weights_impl(shape, dtype, device)

    @abstractmethod
    def _get_weights_impl(
        self, shape: torch.Size, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        pass

    def forward(self, outputs, targets):
        loss = self.loss(outputs, targets)
        return self.reducer(
            loss * self.get_weights(loss.shape, loss.dtype, loss.device)
        )


class NormalScalingLoss(LossScaler):
    def _get_weights_impl(
        self, shape: torch.Size, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        return torch.normal(
            1.0, self.loss_scaling_range, shape, dtype=dtype, device=device
        )


class UniformScalingLoss(LossScaler):
    def _get_weights_impl(
        self, shape: torch.Size, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        return torch.zeros(shape, dtype=dtype, device=device).uniform_(
            1 - self.loss_scaling_range, 1 + self.loss_scaling_range
        )


def init_criterion(args):
    if args.criterion == "crossentropy":
        loss = partial(nn.CrossEntropyLoss)
    else:
        raise NotImplementedError(f"Criterion {args.criterion} not implemented")

    # Default reducer is MeanReducer
    reducer = MeanReducer() if args.loss_scaling is None else NoneReducer()

    if args.reduction == "iqr":
        loss = LossIQR(loss(reduction="none"), reducer=reducer)
    elif args.reduction == "stdmean":
        loss = LossMeanStd(loss(reduction="none"), reducer=reducer)
    else:
        loss = loss(reduction=args.reduction if args.loss_scaling is None else "none")

    if args.loss_scaling is None:
        return loss
    elif args.loss_scaling == "normal-scaling":
        return NormalScalingLoss(loss, MeanReducer())
    elif args.loss_scaling == "uniform-scaling":
        return UniformScalingLoss(loss, MeanReducer())
    else:
        raise NotImplementedError(f"Loss scaling {args.loss_scaling} not implemented")
