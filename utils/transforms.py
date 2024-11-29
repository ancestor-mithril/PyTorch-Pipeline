from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from torchvision.transforms import v2


class BatchHorizontalFlip(nn.Module):
    def __init__(self, p: int = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        flip_mask = (torch.rand(len(x), device=x.device) < self.p).view(-1, 1, 1, 1)
        return torch.where(flip_mask, x.flip(-1), x)


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

    def batch_transforms(self):
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
                self.normalize,
            ]
        )

    def test_cached(self):
        return v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), self.normalize]
        )

    def test_runtime(self):
        return None

    def batch_transforms(self):
        return v2.Compose([
            BatchHorizontalFlip(),
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
        if self.args.cutout:
            fill_value = 0 if self.args.fill is None else self.args.fill
            transforms.append(
                v2.RandomErasing(scale=(0.05, 0.15), value=fill_value, inplace=True)
            )

        transforms.append(self.normalize)
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

    def batch_transforms(self):
        return v2.Compose([
            BatchHorizontalFlip(),
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
