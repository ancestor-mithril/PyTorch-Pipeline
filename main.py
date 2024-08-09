import argparse
import json
import multiprocessing

import torch
from torch.multiprocessing import freeze_support

from utils.trainer import Trainer

if __name__ == '__main__':
    freeze_support()

    parser = argparse.ArgumentParser(description='PyTorch Pipeline')
    parser.add_argument('-device', default='cuda:0', type=str, help='device')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-bs', default=10, type=int, help='batch size')
    parser.add_argument('-epochs', default=200, type=int, help='train epochs')
    parser.add_argument('-es_patience', default=20, type=int, help='early stopping patience')
    parser.add_argument('-dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('-data_path', default='../data', type=str, help='path to dataset')
    parser.add_argument('-scheduler', default='ReduceLROnPlateau', type=str, help='scheduler')
    parser.add_argument('-scheduler_params', default='{}', type=str, help='scheduler_params')
    parser.add_argument('-model', default='preresnet18_c10', type=str, help='model')
    parser.add_argument('-criterion', default='crossentropy', type=str, help='loss')
    parser.add_argument('-reduction', default='mean', type=str, help='loss reduction. Can be mean, iqr, stdmean.')
    parser.add_argument('-fill', default=None, type=float, help='fill value for transformations')
    parser.add_argument('-num_threads', default=None, type=int, help='default number of threads used by pytorch')
    parser.add_argument('-seed', default=3, type=int, help='seed')
    parser.add_argument('--cutout', action='store_true', default=False, help='apply cutout')
    parser.add_argument('--autoaug', action='store_true', default=False, help='apply autoaugment')
    parser.add_argument('--tta', action='store_true', default=False, help='use TTA')
    parser.add_argument('--half', action='store_true', default=False, help='half')
    parser.add_argument('--disable_progress_bar', action='store_true', default=False, help='disable tqdm')
    parser.add_argument('--verbose', action='store_true', default=False, help='verbose')
    parser.add_argument('--stderr', action='store_true', default=False, help='log to stderr instead of stdout')

    args = parser.parse_args()
    args.scheduler_params = json.loads(args.scheduler_params.replace('\'', '"'))

    if args.num_threads is None:
        args.num_threads = multiprocessing.cpu_count() // 2  # use half the available threads
    torch.set_num_threads(args.num_threads)

    Trainer(args).run()

# PYTHONOPTIMIZE=2 python main.py -device cuda:0 -lr 0.001 -bs 16 -epochs 200 -dataset cifar10 -data_path ../data -scheduler ReduceLROnPlateau -scheduler_params "{'mode':'min', 'factor':0.5}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta --half # noqa: E501
# PYTHONOPTIMIZE=2 python main.py -device cuda:0 -lr 0.001 -bs 16 -epochs 200 -dataset cifar10 -data_path ../data -scheduler IncreaseBSOnPlateau -scheduler_params "{'mode':'min', 'factor':2.0, 'max_batch_size': 1000}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta --half # noqa: E501
# python main.py -device cuda:0 -lr 0.001 -bs 16 -epochs 200 -dataset cifar100noisy -data_path ../data -scheduler StepLR -scheduler_params "{'step_size':30, 'gamma': 0.5}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta --half # noqa: E501
# python main.py -device cuda:0 -lr 0.001 -bs 16 -epochs 200 -dataset cifar100noisy -data_path ../data -scheduler StepLR -scheduler_params "{'step_size':30, 'gamma': 0.5}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta --half -reduction iqr # noqa: E501
# python main.py -device cuda:0 -lr 0.001 -bs 16 -epochs 200 -dataset cifar100 -data_path ../data -scheduler StepLR -scheduler_params "{'step_size':30, 'gamma': 0.5}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta --half # noqa: E501
# python main.py -device cuda:0 -lr 0.001 -bs 16 -epochs 200 -dataset cifar100 -data_path ../data -scheduler StepLR -scheduler_params "{'step_size':30, 'gamma': 0.5}" -model preresnet18_c10 -fill 0.5 --cutout --autoaug --tta --half -reduction iqr # noqa: E501
