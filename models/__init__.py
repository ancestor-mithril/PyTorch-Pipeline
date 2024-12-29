import os
from typing import Type, Callable, Any

from torch import nn

from utils.logger import get_logger
from .cnn import CNN_MNIST
from .lenet import LeNet_MNIST
from .preresnet import PreActResNet18_C10


def get_bn_features(bn_layer) -> Any:
    return bn_layer.num_features


def create_identity(new: Type, args: Any) -> nn.Module:
    return new()


def create_norm(new: Type, args: Any) -> nn.Module:
    return new(args)


def replace_layers(model: nn.Module, old: Type, new: Type, furbish_old: Callable[[nn.Module], Any],
                   create_new: Callable[[Type, Any], nn.Module]):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, old, new, furbish_old, create_new)

        if isinstance(module, old):
            args = furbish_old(module)
            setattr(model, n, create_new(new, args))


def replace_bn(model: nn.Module, norm_type: str):
    if norm_type == 'Identity':
        create_fn = create_identity
    else:
        create_fn = create_norm
    norm_type = getattr(nn, norm_type)
    replace_layers(model, nn.BatchNorm2d, norm_type, get_bn_features, create_fn)


def init_model(args, num_classes) -> nn.Module:
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
