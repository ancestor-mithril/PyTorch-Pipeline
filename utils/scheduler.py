from bs_scheduler import IncreaseBSOnPlateau, StepBS, ExponentialBS, PolynomialBS, CosineAnnealingBS, \
    CosineAnnealingBSWithWarmRestarts, CyclicBS, OneCycleBS
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR, PolynomialLR, CosineAnnealingLR, \
    CosineAnnealingWarmRestarts, CyclicLR, OneCycleLR
from torch.utils.data import DataLoader


def init_scheduler(args, optimizer: Optimizer, train_loader: DataLoader):  # noqa C901
    if args.scheduler == 'IncreaseBSOnPlateau':
        # "{'mode':'min', 'factor':2.0, 'max_batch_size': 1000}"

        assert 'factor' in args.scheduler_params
        assert 'max_batch_size' in args.scheduler_params
        scheduler = IncreaseBSOnPlateau(train_loader, **args.scheduler_params)
    elif args.scheduler == 'ReduceLROnPlateau':
        # "{'mode':'min', 'factor':0.5}"

        assert 'factor' in args.scheduler_params
        scheduler = ReduceLROnPlateau(optimizer, **args.scheduler_params)
    elif args.scheduler == 'StepBS':
        # "{'step_size':30, 'gamma': 2.0, 'max_batch_size': 1000}"

        assert 'step_size' in args.scheduler_params
        assert 'max_batch_size' in args.scheduler_params
        scheduler = StepBS(train_loader, **args.scheduler_params)
    elif args.scheduler == 'StepLR':
        # "{'step_size':30, 'gamma': 2.0}"

        assert 'step_size' in args.scheduler_params
        scheduler = StepLR(optimizer, **args.scheduler_params)

    elif args.scheduler == 'ExponentialBS':
        # "{'gamma':1.1, 'max_batch_size': 1000}"

        assert 'gamma' in args.scheduler_params
        assert 'max_batch_size' in args.scheduler_params
        scheduler = ExponentialBS(train_loader, **args.scheduler_params)

    elif args.scheduler == 'ExponentialLR':
        # "{'gamma':0.9}"

        assert 'gamma' in args.scheduler_params
        scheduler = ExponentialLR(optimizer, **args.scheduler_params)

    elif args.scheduler == 'PolynomialBS ':
        # "{'total_iters':200, 'max_batch_size': 1000}"

        assert 'total_iters' in args.scheduler_params
        assert 'max_batch_size' in args.scheduler_params
        scheduler = PolynomialBS(train_loader, **args.scheduler_params)

    elif args.scheduler == 'PolynomialLR':
        # "{'total_iters':200}"

        assert 'total_iters' in args.scheduler_params
        scheduler = PolynomialLR(optimizer, **args.scheduler_params)

    elif args.scheduler == 'CosineAnnealingBS ':
        # "{'total_iters':200, 'max_batch_size': 1000}"

        assert 'total_iters' in args.scheduler_params
        assert 'max_batch_size' in args.scheduler_params
        scheduler = CosineAnnealingBS(train_loader, **args.scheduler_params)

    elif args.scheduler == 'CosineAnnealingLR':
        # "{'T_max':200}"

        assert 'T_max' in args.scheduler_params
        scheduler = CosineAnnealingLR(optimizer, **args.scheduler_params)

    elif args.scheduler == 'CosineAnnealingBSWithWarmRestarts  ':
        # "{'t_0':100, 'factor':1, 'max_batch_size': 1000}"

        assert 't_0' in args.scheduler_params
        assert 'factor' in args.scheduler_params
        assert 'max_batch_size' in args.scheduler_params
        scheduler = CosineAnnealingBSWithWarmRestarts(train_loader, **args.scheduler_params)

    elif args.scheduler == 'CosineAnnealingWarmRestarts ':
        # "{'T_0':100, 'T_mult': 1}"

        assert 'T_0' in args.scheduler_params
        assert 'T_mult' in args.scheduler_params
        scheduler = CosineAnnealingWarmRestarts(optimizer, **args.scheduler_params)

    elif args.scheduler == 'CyclicBS':
        # "{'min_batch_size':10, 'base_batch_size': 500, 'step_size_down': 20, 'mode': 'triangular2', 'max_batch_size': 1000}"

        assert 'min_batch_size' in args.scheduler_params
        assert 'base_batch_size' in args.scheduler_params
        assert 'step_size_down' in args.scheduler_params
        assert 'mode' in args.scheduler_params
        assert 'max_batch_size' in args.scheduler_params
        scheduler = CyclicBS(train_loader, **args.scheduler_params)

    elif args.scheduler == 'CyclicLR':
        # "{'base_lr':0.0001, 'max_lr': 0.01, 'step_size_up': 20, 'mode': 'triangular2'}"

        assert 'base_lr' in args.scheduler_params
        assert 'max_lr' in args.scheduler_params
        assert 'step_size_up' in args.scheduler_params
        assert 'mode' in args.scheduler_params
        scheduler = CyclicLR(optimizer, **args.scheduler_params)

    elif args.scheduler == 'OneCycleBS':
        # "{'total_steps':200, 'base_batch_size': 300, 'min_batch_size': 10, 'max_batch_size': 1000}"

        assert 'total_steps' in args.scheduler_params
        assert 'base_batch_size' in args.scheduler_params
        assert 'min_batch_size' in args.scheduler_params
        assert 'max_batch_size' in args.scheduler_params
        scheduler = OneCycleBS(train_loader, **args.scheduler_params)

    elif args.scheduler == 'OneCycleLR':
        # "{'total_steps':200, 'max_lr': 0.01}"

        assert 'total_steps' in args.scheduler_params
        assert 'max_lr' in args.scheduler_params
        scheduler = OneCycleLR(optimizer, **args.scheduler_params)
    else:
        raise NotImplementedError(f'Scheduler {args.scheduler} not implemented')
    return scheduler
