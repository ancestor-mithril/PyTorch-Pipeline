from functools import partial

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

from .transforms import DatasetTransforms, init_transforms


def identity(x):
    return x


class CachedDataset(Dataset):
    def __init__(self, dataset, transforms=None, num_classes=None, cache=True):
        if cache:
            self.data = tuple([x for x in dataset])
        else:
            self.data = dataset
        self.transforms = transforms
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        if self.transforms is not None:
            image, label = self.data[i]
            return self.transforms(image), label
        return self.data[i]


def init_dataset(args):
    if args.dataset == 'cifar10':
        dataset_fn = partial(CIFAR10, root=args.data_path, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataset_fn = partial(CIFAR100, root=args.data_path, download=True)
        num_classes = 100
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not implemented')

    transforms = init_transforms(args)

    cache_train_dataset = True if not hasattr(args, 'cache_train_dataset') else args.cache_train_dataset
    cache_test_dataset = True if not hasattr(args, 'cache_test_dataset') else args.cache_test_dataset

    train_dataset = dataset_fn(train=True, transform=transforms.train_cached())
    train_dataset = CachedDataset(train_dataset, transforms=transforms.train_runtime(), num_classes=num_classes,
                                  cache=cache_train_dataset)

    test_dataset = dataset_fn(train=False, transform=transforms.test_cached())
    test_dataset = CachedDataset(test_dataset, transforms=transforms.train_runtime(), num_classes=num_classes,
                                 cache=cache_test_dataset)

    return train_dataset, test_dataset


def init_loaders(args, train_dataset: CachedDataset, test_dataset: CachedDataset, pin_memory):
    shuffle_train = True if not hasattr(args, 'shuffle_train') else args.shuffle_train
    num_workers = 0 if not hasattr(args, 'num_workers') else args.num_workers
    drop_last = True if not hasattr(args, 'drop_last') else args.drop_last

    bs_val = 500 if not hasattr(args, 'bs_val') else args.bs_val
    num_workers_val = 0 if not hasattr(args, 'num_workers_val') else args.num_workers_val

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=shuffle_train, num_workers=num_workers,
                              drop_last=drop_last, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=bs_val, shuffle=False, num_workers=num_workers_val,
                             pin_memory=pin_memory)
    return train_loader, test_loader
