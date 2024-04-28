# pip install git+https://github.com/ancestor-mithril/bs_scheduler.git@master
from bs_scheduler import IncreaseBSOnPlateau
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


def init_scheduler(args, optimizer: Optimizer, train_loader: DataLoader):
    if args.scheduler == 'IncreaseBSOnPlateau':
        scheduler = IncreaseBSOnPlateau(train_loader, **args.scheduler_params)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **args.scheduler_params)
    else:
        raise NotImplementedError(f'Scheduler {args.scheduler} not implemented')
    return scheduler
