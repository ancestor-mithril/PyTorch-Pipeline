from abc import ABC, abstractmethod
from typing import Sequence, Callable

import torch
from torch import nn, Tensor
from torchvision.transforms import v2


class BatchHorizontalFlip(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        flip_mask = (torch.rand(len(x), device=x.device) < self.p).view(-1, 1, 1, 1)
        return torch.where(flip_mask, x.flip(-1), x)


class AlternativeHorizontalFlip(nn.Module):
    # Inspired from https://github.com/KellerJordan/cifar10-airbench
    # https://arxiv.org/abs/2404.00498: 94% on CIFAR-10 in 3.29 Seconds on a Single GPU
    def __init__(self):
        super().__init__()
        self.epoch: int = 0

    @staticmethod
    def init(x: Tensor) -> Tensor:
        return BatchHorizontalFlip(p=0.5)(x)

    def step(self):
        self.epoch += 1

    def forward(self, x: Tensor) -> Tensor:
        if self.epoch % 2 == 1:
            return x.flip(-1)
        return x


class StepCompose(v2.Compose):
    def __init__(self, transforms: Sequence[Callable]):
        super().__init__(transforms)
        self.need_step = [x for x in self.transforms if hasattr(x, "step")]

    def init(self, x: Tensor) -> Tensor:
        for transform in self.transforms:
            if hasattr(transform, "init"):
                x = transform.init(x)
        return x

    def step(self):
        for transform in self.transforms:
            if hasattr(transform, "step"):
                transform.step()


class DatasetTransforms(ABC):
    @abstractmethod
    def train_cached(self):
        pass

    @abstractmethod
    def train_runtime(self):
        pass

    @abstractmethod
    def test_cached(self):
        pass

    @abstractmethod
    def test_runtime(self):
        pass

    def batch_transforms_device(self):
        return None

    def batch_transforms_cpu(self):
        return None


class MNISTTransforms(DatasetTransforms):
    def __init__(self, args):
        self.args = args
        self.normalize = v2.Normalize((0.1307,), (0.3081,), inplace=True)

    def train_cached(self):
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def train_runtime(self):
        return v2.Compose(
            [
                v2.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                v2.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        )

    def test_cached(self):
        return v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), self.normalize]
        )

    def test_runtime(self):
        return None

    def batch_transforms_cpu(self):
        # TODO: Use this in collate_fn
        return None

    def batch_transforms_device(self):
        return StepCompose([
            AlternativeHorizontalFlip(), # TODO: move on cpu
            self.normalize
        ])


class CifarTransforms(DatasetTransforms):
    def __init__(self, args):
        self.args = args
        self.normalize = v2.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True
        )

    def train_cached(self):
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

    def train_runtime(self):
        transforms = [
            v2.RandomCrop(
                32, padding=4, fill=0 if self.args.fill is None else self.args.fill
            ),
        ]

        if self.args.autoaug:
            transforms.append(
                v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10, fill=self.args.fill)
            )

        transforms = v2.Compose(transforms)
        return transforms

    def test_cached(self):
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.normalize,
            ]
        )

    def test_runtime(self):
        return None

    def create_cutout(self):
        fill_value = 0 if self.args.fill is None else self.args.fill
        return v2.RandomErasing(scale=(0.05, 0.15), value=fill_value, inplace=True)

    def batch_transforms_cpu(self):
        if self.args.cutout:
            return StepCompose([
                self.create_cutout()
            ])
        return None

    def batch_transforms_device(self):
        return StepCompose([
            AlternativeHorizontalFlip(),
            self.normalize
        ])


class FashionMNISTTransforms(CifarTransforms):
    def __init__(self, args):
        super().__init__(args)
        self.normalize = v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # FashionMNIST must not have inplace normalize, otherwise it fails because it is a single channel image


def init_transforms(args) -> DatasetTransforms:
    if args.dataset in ("cifar10", "cifar100", "cifar100noisy"):
        return CifarTransforms(args)
    if args.dataset in ("FashionMNIST",):
        return FashionMNISTTransforms(args)
    if args.dataset in ("MNIST", "DirtyMNIST"):
        return MNISTTransforms(args)
    raise NotImplementedError(f"Transforms not implemented for {args.dataset}")
