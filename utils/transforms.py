from abc import ABC, abstractmethod

import torch
from torchvision.transforms import v2


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


class CifarTransforms(DatasetTransforms):
    def __init__(self, args):
        self.args = args

    def train_cached(self):
        return v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        ])

    def train_runtime(self):
        transforms = [
            v2.RandomCrop(32, padding=4, fill=0 if self.args.fill is None else self.args.fill),
            v2.RandomHorizontalFlip(),
        ]

        if self.args.autoaug:
            transforms.append(v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10, fill=self.args.fill))
        if self.args.cutout:
            fill_value = 0 if self.args.fill is None else self.args.fill
            transforms.append(v2.RandomErasing(scale=(0.05, 0.15), value=fill_value, inplace=True))

        transforms.append(v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transforms = v2.Compose(transforms)
        return transforms

    def test_cached(self):
        return v2.Compose([
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def test_runtime(self):
        return None


def init_transforms(args) -> DatasetTransforms:
    if args.dataset in ('cifar10', 'cifar100', 'FashionMNIST'):
        return CifarTransforms(args)
    raise NotImplementedError(f"Transforms not implemented for {args.dataset}")
