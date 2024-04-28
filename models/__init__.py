from .preresnet import PreActResNet18_C10


def init_model(args, num_classes):
    if args.model == 'preresnet18_c10':
        model = PreActResNet18_C10(num_classes)
    else:
        raise NotImplementedError(f'Model {args.model} not implemented')

    return model
