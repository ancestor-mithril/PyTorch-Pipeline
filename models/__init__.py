import os
from typing import Type

import torch.nn

from utils.logger import get_logger
from .cnn import CNN_MNIST
from .lenet import LeNet_MNIST
from .preresnet import PreActResNet18_C10


def replace_layers(model: torch.nn.Module, old: Type, new: Type):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old, new)

        if isinstance(module, old):
            setattr(model, n, new())


def replace_bn(model: torch.nn.Module, norm_type: str):
    norm_type = getattr(torch.nn, norm_type)
    replace_layers(model, torch.nn.BatchNorm2d, norm_type)


def init_model(args, num_classes) -> torch.nn.Module:
    if args.model == "preresnet18_c10":
        model = PreActResNet18_C10(num_classes)
    elif args.model == "lenet_mnist":
        model = LeNet_MNIST(num_classes)
    elif args.model == "cnn_mnist":
        model = CNN_MNIST(num_classes)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")

    norm_type = os.getenv('NN_norm_type', None)
    if norm_type is not None:
        get_logger().log_both(f'Using {norm_type} instead of BatchNorm2d')
        replace_bn(model, norm_type)

    return model
