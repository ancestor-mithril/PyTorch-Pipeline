import os
import re
from datetime import datetime
from timeit import default_timer

import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2.functional import hflip, vflip  # noqa: F401
from tqdm import tqdm

from models import init_model
from utils.dataset import init_dataset, init_loaders
from utils.loss import init_criterion
from utils.optimizer import init_optimizer
from utils.scheduler import init_scheduler


class Trainer:
    def __init__(self, args):
        self.args = args

        self.device = torch.device(args.device)
        print(f'Using {self.device}')

        pin_memory = False
        if self.device.type == 'cuda':
            cudnn.benchmark = True
            pin_memory = True

        self.train_dataset, self.test_dataset = init_dataset(args)
        self.train_loader, self.test_loader = init_loaders(args, self.train_dataset, self.test_dataset, pin_memory)

        self.model = init_model(args, self.train_dataset.num_classes).to(self.device)
        self.model = torch.jit.script(self.model)

        self.criterion = init_criterion(args)
        self.optimizer = init_optimizer(args, self.model.parameters())

        self.scheduler = init_scheduler(args, self.optimizer, self.train_loader)

        self.logdir = self.init_logdir()
        self.writer = SummaryWriter(log_dir=f'runs/{self.logdir}')
        self.best_metric = 0.0

    def init_logdir(self):
        params = self.args.__dict__
        params.pop('device')
        params.pop('data_path')

        params = [f'{k}_{v}' for k, v in params.items()]
        params = [re.sub(r'[^a-zA-Z0-9_.-]', '_', x.replace(' ', '')) for x in params]

        now = datetime.now()
        logdir = os.path.join(now.strftime('%y-%m-%d'), now.strftime('%H-%M-%S'), *params)
        return logdir

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def get_bs(self):
        return self.train_loader.batch_sampler.batch_size

    def run(self):
        # TODO: Implement return time instead of printing
        epochs = list(range(self.args.epochs))
        total_training_time = 0
        try:
            with tqdm(epochs) as tbar:
                for epoch in tbar:
                    start = default_timer()
                    metrics = self.train()
                    end = default_timer()
                    total_training_time += end - start

                    metrics.update(self.val())

                    self.post_epoch(metrics)
                    self.write_metrics(epoch, metrics, total_training_time)
                    tbar.set_description(self.epoch_description(metrics))
        except KeyboardInterrupt:
            pass
        with open("results.txt", "a") as f:
            f.write(f'{self.logdir} -> {self.best_metric}\n')
        print("Best:", self.best_metric)

    def train(self):
        self.model.train()
        correct = 0
        total = 0
        total_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.args.half):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {
            'Train/Accuracy': 100. * correct / total,
            'Train/Loss': total_loss
        }

    @torch.inference_mode()
    def val(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            with torch.autocast(self.device.type, enabled=self.args.half):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                if self.args.tta:
                    outputs += self.model(hflip(inputs))
                    # TODO: Compare partial TTA with full TTA
                    # outputs += self.model(vflip(inputs))
                    # outputs += self.model(vflip(hflip(inputs)))

            total_loss += loss.item()
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {
            'Val/Accuracy': 100. * correct / total,
            'Val/Loss': total_loss
        }

    def save_checkpoint(self, metrics):
        optimized_metric = 'Val/Accuracy' if not hasattr(
            self.args, 'optimized_metric') else self.args.optimized_metric
        optimized_metric = metrics[optimized_metric]
        if optimized_metric > self.best_metric:
            self.best_metric = optimized_metric
            # TODO: Save model if saving is enabled

    def empty_cache(self):
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def write_metrics(self, epoch, metrics, training_time):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, epoch)
        self.writer.add_scalar('Train/Time', training_time, epoch)

        self.writer.add_scalar('Trainer/Learning Rate', self.get_lr(), epoch)
        self.writer.add_scalar('Trainer/Batch Size', self.get_bs(), epoch)

    def scheduler_step(self, metrics):
        if self.scheduler is None:
            return

        if type(self.scheduler).__name__ in ('ReduceLROnPlateau', 'IncreaseBSOnPlateau'):
            scheduler_metric = 'Val/Accuracy' if not hasattr(
                self.args, 'scheduler_metric') else self.args.scheduler_metric
            if type(self.scheduler).__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(metric=metrics[scheduler_metric])
                # TODO: keep the same name
            else:
                self.scheduler.step(metrics[scheduler_metric])
        else:
            self.scheduler.step()

    def post_epoch(self, metrics: dict):
        self.empty_cache()
        self.save_checkpoint(metrics)
        self.scheduler_step(metrics)
        self.early_stopping(metrics)

    def epoch_description(self, metrics):
        train_acc = round(metrics["Train/Accuracy"], 2)
        val_acc = round(metrics["Val/Accuracy"], 2)
        best = round(self.best_metric, 2)
        return f'Train: {train_acc}, Val: {val_acc}, Best: {best}'

    def early_stopping(self, metrics):
        # TODO: Implement early stopping
        pass
