import os.path
from urllib.request import urlretrieve
from typing import Optional, Callable

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100

from utils.logger import get_logger


def download_noisy_labels(url: str, file_path: str):
    get_logger().log(f"Downloading {file_path} from {url}")
    urlretrieve(url, file_path)


class CIFAR100_noisy_fine(Dataset):
    """
    See https://github.com/UCSC-REAL/cifar-10-100n, https://www.noisylabels.com/ and `Learning with Noisy Labels
    Revisited: A Study Using Real-World Human Annotations`.
    """

    def __init__(self, root: str, train: bool, transform: Optional[Callable], download: bool):
        cifar100 = CIFAR100(root=root, train=train, transform=transform, download=download)
        data, targets = tuple(zip(*cifar100))

        if train:
            noisy_label_file = os.path.join(root, 'CIFAR-100-noisy.npz')
            if not os.path.isfile(noisy_label_file):
                if not download:
                    raise FileNotFoundError(f"{type(self).__name__} need {noisy_label_file} to be used!")
                download_noisy_labels('https://github.com/ancestor-mithril/PyTorch-Pipeline'
                                      '/raw/master/data-repository/CIFAR-100-noisy.npz',
                                      noisy_label_file)

            noise_file = np.load(noisy_label_file)
            if not np.array_equal(noise_file['clean_label'], targets):
                raise RuntimeError("Clean labels do not match!")
            targets = noise_file['noisy_label']

        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i: int):
        return self.data[i], self.targets[i]
