import os
import re
from datetime import datetime
from functools import cached_property

import torch
from timed_decorator.simple_timed import timed
from torch import GradScaler, Tensor, nn
from torch.backends import cudnn
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2.functional import hflip, vflip  # noqa: F401
from tqdm import tqdm

from models import init_model
from utils.dataset import init_dataset, init_loaders
from utils.early_stopping import init_early_stopping
from utils.logger import init_logger
from utils.loss import init_criterion
from utils.optimizer import init_optimizer
from utils.scheduler import init_scheduler
from utils.utils import seed_everything, try_optimize


class Trainer:
    def __init__(self, args):
        self.args = args

        self.logdir = self.init_logdir()
        self.logger = init_logger(self.logdir, args.verbose, args.stderr)
        self.logger.log_both(args)

        seed_everything(args.seed)

        self.device = torch.device(args.device)
        self.logger.log_both(f"Using {self.device}")

        if self.device.type == "cuda":
            cudnn.benchmark = True
            pin_memory = True
            enable_grad_scaler = self.args.half
        else:
            self.args.half = False
            enable_grad_scaler = False
            # half is slower on cpu. does not work on mps
            pin_memory = False

        self.scaler = GradScaler(self.device.type, enabled=enable_grad_scaler)

        self.train_dataset, self.test_dataset = init_dataset(args)
        self.batch_transforms_cpu = self.train_dataset.batch_transforms_cpu
        self.batch_transforms_device = self.train_dataset.batch_transforms_device
        self.train_loader, self.test_loader = init_loaders(
            args, self.train_dataset, self.test_dataset, pin_memory
        )

        self.model = init_model(args, self.train_dataset.num_classes).to(self.device)

        self.criterion = init_criterion(args).to(self.device)
        self.optimizer = init_optimizer(args, self.model.parameters())

        self.scheduler = init_scheduler(args, self.optimizer, self.train_loader)
        self.early_stopper = init_early_stopping(args)

        self.writer = SummaryWriter(log_dir=self.logdir)
        self.best_metric = 0.0

        self.optimize()

    def optimize(self):
        self.logger.log_both("Optimizing model")
        self.model = try_optimize(self.model)

        self.logger.log_both("Optimizing criterion")
        self.criterion = try_optimize(self.criterion)

        self.logger.log_both("Optimizing optimizer")
        self.optimizer = try_optimize(
            self.optimizer
        )  # trace/script does not work, compile works instead (on UNIX)

    @cached_property
    def scheduler_metric(self):
        return (
            "Train/Loss"
            if not hasattr(self.args, "scheduler_metric")
            else self.args.scheduler_metric
        )

    @cached_property
    def optimized_metric(self):
        return (
            "Val/Accuracy"
            if not hasattr(self.args, "optimized_metric")
            else self.args.optimized_metric
        )

    @cached_property
    def es_metric(self):
        return (
            "Train/Loss" if not hasattr(self.args, "es_metric") else self.args.es_metric
        )

    def init_logdir(self):
        params = {**self.args.__dict__}
        params.pop("device")
        params.pop("data_path")
        params.pop("disable_progress_bar")
        params.pop("verbose")
        params.pop("stderr")
        params.pop("half")
        params.pop("num_threads")

        # removed for no info, add again if needed
        params.pop("criterion")

        # FIXME: Adding scheduler and scheduler params means changing plot creator script
        no_keys = [
            "dataset",
            # 'scheduler',
            # 'scheduler_params',
            "model",
            "reduction",
        ]

        params = [f"{k}_{v}" if k not in no_keys else f"{v}" for k, v in params.items()]
        params = [re.sub(r"[^a-zA-Z0-9_.-]", "_", x.replace(" ", "")) for x in params]
        params = [re.sub("_+", "_", x) for x in params]

        now = datetime.now()
        logdir = os.path.join(
            "runs", now.strftime("%y-%m-%d"), now.strftime("%H-%M-%S"), *params
        )
        return logdir

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_bs(self):
        return self.train_loader.batch_sampler.batch_size

    def run(self):
        epochs = list(range(self.args.epochs))
        total_training_time = 0
        try:
            with tqdm(epochs, disable=self.args.disable_progress_bar) as tbar:
                for epoch in tbar:
                    metrics, elapsed = self.train()
                    total_training_time += elapsed / 1e9

                    metrics.update(self.val())

                    self.post_epoch(metrics)
                    self.write_metrics(epoch, metrics, total_training_time)
                    self.update_tbar(tbar, metrics, epoch)
                    if self.early_stopping(metrics):
                        self.logger.log_both("Early stopping")
                        break
        except KeyboardInterrupt:
            pass
        with open("results.txt", "a") as f:
            f.write(f"{self.logdir} -> {self.best_metric}\n")
        self.logger.log_both(f"Best: {self.best_metric}, after {epoch} epochs")

    @timed(stdout=False, return_time=True)
    def train(self):
        self.model.train()
        correct = 0
        total = 0
        total_loss = 0.0

        for inputs, targets in self.train_loader:
            inputs = self.prepare_inputs(inputs, self.device)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.args.half):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            self.scaler.scale(loss).backward()
            self.maybe_clip()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {"Train/Accuracy": 100.0 * correct / total, "Train/Loss": total_loss}

    @torch.inference_mode()
    def val(self):
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        for inputs, targets in self.test_loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.autocast(self.device.type, enabled=self.args.half):
                if self.args.tta:  # This is flip tta
                    combined = torch.cat([
                        inputs, hflip(inputs)
                    ], dim=0)
                    outputs = sum(self.model(combined).split(len(inputs)))
                else:
                    outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item()
            predicted = outputs.argmax(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        return {"Val/Accuracy": 100.0 * correct / total, "Val/Loss": total_loss}

    def save_checkpoint(self, metrics):
        optimized_metric = metrics[self.optimized_metric]
        if optimized_metric > self.best_metric:
            self.best_metric = optimized_metric
            # TODO: Save model if saving is enabled

    def empty_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def write_metrics(self, epoch, metrics, training_time):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, epoch)
        self.writer.add_scalar("Train/Time", training_time, epoch)

        self.writer.add_scalar("Trainer/Learning Rate", self.get_lr(), epoch)
        self.writer.add_scalar("Trainer/Batch Size", self.get_bs(), epoch)

    def scheduler_step(self, metrics):
        if self.scheduler is None:
            return

        if type(self.scheduler).__name__ in (
            "ReduceLROnPlateau",
            "IncreaseBSOnPlateau",
        ):
            scheduler_metric = metrics[self.scheduler_metric]

            if type(self.scheduler).__name__ == "IncreaseBSOnPlateau":
                self.scheduler.step(metric=scheduler_metric)
                # TODO: keep the same name
            else:
                self.scheduler.step(scheduler_metric)
        else:
            self.scheduler.step()

    def post_epoch(self, metrics: dict):
        self.empty_cache()
        self.save_checkpoint(metrics)
        self.scheduler_step(metrics)

    def epoch_description(self, metrics):
        train_acc = round(metrics["Train/Accuracy"], 2)
        val_acc = round(metrics["Val/Accuracy"], 2)
        best = round(self.best_metric, 2)
        # DEBUG
        if hasattr(self.criterion, "progress_tracker"):
            progress = round(
                torch.mean(torch.Tensor(self.criterion.progress_tracker)).item(), 2
            )
            self.criterion.progress_tracker = []
            return f"Train: {train_acc}, Val: {val_acc}, Best: {best}, Progress: {progress}"
        return f"Train: {train_acc}, Val: {val_acc}, Best: {best}"

    def early_stopping(self, metrics):
        es_metric = metrics[self.es_metric]
        return self.early_stopper.step(es_metric)

    def update_tbar(self, tbar, metrics, epoch):
        description = self.epoch_description(metrics)
        self.logger.log(f"Epoch: {epoch},", description, to_console=self.args.disable_progress_bar)
        if not self.args.disable_progress_bar:
            tbar.set_description(description)

    def maybe_clip(self):
        if self.args.clip_value is not None:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args.clip_value)

    def prepare_inputs(self, x: Tensor, device: torch.device) -> Tensor:
        if self.batch_transforms_cpu is not None:
            x = self.batch_transforms_cpu(x)
        x = x.to(device, non_blocking=True)
        if self.batch_transforms_device is not None:
            x = self.batch_transforms_device(x)
        return x
