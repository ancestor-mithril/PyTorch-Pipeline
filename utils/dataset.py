from functools import partial
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, default_collate
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, MNIST

from .datasets import CIFAR100_noisy_fine
from .transforms import init_transforms, StepCompose


def identity(x):
    return x


class CachedDataset(Dataset):
    def __init__(self, dataset, transforms=None, num_classes=None, batch_transforms_cpu: Optional[StepCompose] = None,
                 batch_transforms_device: Optional[StepCompose] = None):
        self.transforms = transforms
        self.num_classes = num_classes
        self.batch_transforms_cpu = batch_transforms_cpu
        self.batch_transforms_device = batch_transforms_device
        self.data, self.targets = self.cache_dataset(dataset)

    def cache_dataset(self, dataset):
        data = []
        targets = []
        for x, y in dataset:
            data.append(x)
            targets.append(y)
        data = torch.stack(data)
        if self.batch_transforms_cpu is not None:
            data = self.batch_transforms_cpu.init(data)
        if self.batch_transforms_device is not None:
            data = self.batch_transforms_device.init(data)
        return tuple(data), tuple(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        if self.transforms is not None:
            data = self.transforms(data)
        return data, self.targets[i]


def init_dataset(args):
    if args.dataset == "cifar10":
        dataset_fn = partial(CIFAR10, root=args.data_path, download=True)
        num_classes = 10
    elif args.dataset == "cifar100":
        dataset_fn = partial(CIFAR100, root=args.data_path, download=True)
        num_classes = 100
    elif args.dataset == "cifar100noisy":
        # Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations
        # https://github.com/UCSC-REAL/cifar-10-100n
        # https://www.noisylabels.com/
        dataset_fn = partial(CIFAR100_noisy_fine, root=args.data_path, download=True)
        num_classes = 100
    elif args.dataset == "FashionMNIST":
        dataset_fn = partial(FashionMNIST, root=args.data_path, download=True)
        num_classes = 10
    elif args.dataset == "MNIST":
        dataset_fn = partial(MNIST, root=args.data_path, download=True)
        num_classes = 10
    elif args.dataset == "DirtyMNIST":
        # Deterministic Neural Networks with Appropriate Inductive Biases Capture Epistemic and Aleatoric Uncertainty
        # https://github.com/omegafragger/DDU
        # Dataset: https://github.com/BlackHC/ddu_dirty_mnist
        from ddu_dirty_mnist import DirtyMNIST

        dataset_fn = partial(DirtyMNIST, root=args.data_path, download=True)
        num_classes = 10
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    transforms = init_transforms(args)

    train_dataset = dataset_fn(train=True, transform=transforms.train_cached())
    train_dataset = CachedDataset(
        train_dataset,
        transforms=transforms.train_runtime(),
        num_classes=num_classes,
        batch_transforms_cpu=transforms.batch_transforms_cpu(),
        batch_transforms_device=transforms.batch_transforms_device(),
    )

    test_dataset = dataset_fn(train=False, transform=transforms.test_cached())
    test_dataset = CachedDataset(
        test_dataset,
        transforms=transforms.test_runtime(),
        num_classes=num_classes,
    )

    return train_dataset, test_dataset


def custom_collate(cpu_transforms: Callable):
    def collator(batch):
        data, labels = default_collate(batch)
        return cpu_transforms(data), labels

    return collator


def init_loaders(
        args, train_dataset: CachedDataset, test_dataset: CachedDataset, pin_memory
):
    shuffle_train = True if not hasattr(args, "shuffle_train") else args.shuffle_train
    num_workers = 0 if not hasattr(args, "num_workers") else args.num_workers
    drop_last = True if not hasattr(args, "drop_last") else args.drop_last

    bs_val = 500 if not hasattr(args, "bs_val") else args.bs_val
    num_workers_val = (
        0 if not hasattr(args, "num_workers_val") else args.num_workers_val
    )

    train_collate_fn = default_collate
    if train_dataset.batch_transforms_cpu is not None:
        train_collate_fn = custom_collate(train_dataset.batch_transforms_cpu)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        collate_fn=train_collate_fn,
    )

    test_collate_fn = default_collate
    if test_dataset.batch_transforms_cpu is not None:
        test_collate_fn = custom_collate(test_dataset.batch_transforms_cpu)

    test_loader = DataLoader(
        test_dataset,
        batch_size=bs_val,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=pin_memory,
        collate_fn=test_collate_fn,
    )
    return train_loader, test_loader
