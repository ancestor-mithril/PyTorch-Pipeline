from torch.optim.lr_scheduler import ReduceLROnPlateau


def init_scheduler(args, optimizer, train_loader):
    if args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, **args.scheduler_params)
    else:
        raise NotImplementedError(f'Scheduler {args.scheduler} not implemented')
    return scheduler
