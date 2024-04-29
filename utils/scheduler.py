from bs_scheduler import IncreaseBSOnPlateau, StepBS
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader


def init_scheduler(args, optimizer: Optimizer, train_loader: DataLoader):
    if args.scheduler == 'IncreaseBSOnPlateau':
        # "{'mode':'min', 'factor':2.0, 'max_batch_size': 1000}"
        scheduler = IncreaseBSOnPlateau(train_loader, **args.scheduler_params)
    elif args.scheduler == 'ReduceLROnPlateau':
        # "{'mode':'min', 'factor':0.5}"
        scheduler = ReduceLROnPlateau(optimizer, **args.scheduler_params)
    elif args.scheduler == 'StepBS':
        # "{'step_size':30, 'gamma': 2.0, 'max_batch_size': 1000}"
        scheduler = StepBS(train_loader, **args.scheduler_params)
    elif args.scheduler == 'StepLR':
        # "{'step_size':30, 'gamma': 2.0}"
        scheduler = StepLR(optimizer, **args.scheduler_params)
    else:
        raise NotImplementedError(f'Scheduler {args.scheduler} not implemented')
    return scheduler
