from .cnn import CNN_MNIST
from .lenet import LeNet_MNIST
from .preresnet import PreActResNet18_C10


def init_model(args, num_classes):
    if args.model == 'preresnet18_c10':
        model = PreActResNet18_C10(num_classes)
    elif args.model == 'lenet_mnist':
        model = LeNet_MNIST(num_classes)
    elif args.model == 'cnn_mnist':
        model = CNN_MNIST(num_classes)
    else:
        raise NotImplementedError(f'Model {args.model} not implemented')

    return model
