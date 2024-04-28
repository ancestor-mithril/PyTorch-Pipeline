import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

import torchvision
import torchvision.transforms.v2 as v2
from torchvision.transforms.v2.functional import vflip, hflip
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import argparse

from models.preresnet import PreActResNet18_C10

best_acc = 0.0


class CachedDataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.data = tuple([x for x in dataset])
        self.transforms = transforms if transforms is not None else nn.Identity()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image, label = self.data[i]
        return self.transforms(image), label


def train(model, loader, criterion, optimizer, device, half):
    model.train()
    correct = 0
    total = 0
    total_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return 100. * correct / total, total_loss


@torch.inference_mode()
def val(model, loader, criterion, device, half, tta):
    global best_acc
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=half):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if tta:
                outputs += model(hflip(inputs))

        total_loss += loss.item()
        predicted = outputs.argmax(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return acc, total_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('-device', default='cuda:0', type=str, help='device')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('-bs', default=10, type=int, help='batch size')
    parser.add_argument('-epochs', default=200, type=int, help='train epochs')
    parser.add_argument('-dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('-scheduler', default='ReduceLROnPlateau', type=str, help='scheduler')
    parser.add_argument('-scheduler_params', default="{'mode':'min', 'factor':0.5}", type=str, help='scheduler_params')
    parser.add_argument('-fill', default=None, type=float, help='fill value')
    parser.add_argument('-model', default='preresnet', type=str, help='model')
    parser.add_argument('--cutout', action='store_true', default=False, help='apply cutout')
    parser.add_argument('--autoaug', action='store_true', default=False, help='apply autoaugment')
    parser.add_argument('--tta', action='store_true', default=False, help='tta')
    parser.add_argument('--half', action='store_true', default=False, help='half')
    args = parser.parse_args()
    args.scheduler_params = eval(args.scheduler_params)
    print(args)

    cached_transforms = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
    ])
    transform_train = [
        v2.RandomCrop(32, padding=4, fill=0 if args.fill is None else args.fill),
        v2.RandomHorizontalFlip(),
    ]

    if args.autoaug:
        transform_train.append(
            v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10, fill=args.fill)
        )
    if args.cutout:
        transform_train.append(
            v2.RandomErasing(scale=(0.05, 0.15), value=0 if args.fill is None else args.fill, inplace=True)
        )
    transform_train.append(
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    )

    transform_train = v2.Compose(transform_train)

    transform_test = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    device = torch.device(args.device)
    print(device)
    half = False
    pin_memory = False
    if device.type == 'cuda':
        cudnn.benchmark = True
        half = True
        pin_memory = True

    if args.dataset == 'cifar10':
        dataset_fn = CIFAR10
        num_classes = 10
    elif args.dataset == 'cifar100':
        dataset_fn = CIFAR100
        num_classes = 100

    trainset = CachedDataset(dataset_fn(root='./data', train=True, download=True, transform=cached_transforms),
                             transform_train)
    testset = CachedDataset(dataset_fn(root='./data', train=False, download=True, transform=transform_test))

    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=0, drop_last=True,
                             pin_memory=pin_memory)
    testloader = DataLoader(testset, batch_size=500, shuffle=False, num_workers=0, pin_memory=pin_memory)

    if args.model == 'preresnet':
        model_fn = PreActResNet18_C10

    model = model_fn(num_classes).to(device)
    model = torch.jit.script(model)
    # model = torch.compile(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

    if args.scheduler == 'ReduceLROnPlateau':
        scheduler_fn = ReduceLROnPlateau

    scheduler = scheduler_fn(optimizer, **args.scheduler_params)

    try:
        with tqdm(list(range(args.epochs))) as tbar:
            for epoch in tbar:
                train_acc, train_loss = train(model, trainloader, criterion, optimizer, device, args.half)
                val_acc, val_loss = val(model, testloader, criterion, device, args.half, args.tta)
                scheduler.step(val_loss)
                tbar.set_description(
                    f'Train: {round(train_acc, 2)}, Val: {round(val_acc, 2)}, Best: {round(best_acc, 2)}, LR: {scheduler.get_last_lr()}'
                )
    except KeyboardInterrupt:
        pass
    with open("results.txt", "a") as f:
        f.write(str(args) + ' -> ' + str(best_acc) + '\n')
    print("Best:", best_acc)
