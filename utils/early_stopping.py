import numpy as np

from utils.logger import get_logger


class EarlyStopping:
    def __init__(self, mode="min", min_delta=0.0, patience=10, percentage=False):
        self.is_better = None
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0

        if patience == 0:
            self.step = self.fake_step
        else:
            self.init_is_better(mode, percentage)

    @staticmethod
    def fake_step(metrics):
        return False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            get_logger().log_both(
                "None encountered in metrics. Early stopping activated."
            )
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True
        return False

    def init_is_better(self, mode: str, percentage: bool) -> callable:
        if mode not in ("min", "max"):
            raise ValueError("mode " + mode + " is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = self.absolute_min
            if mode == "max":
                self.is_better = self.absolute_max
        else:
            if mode == "min":
                self.min_delta = 1 - self.min_delta / 100
                self.is_better = self.relative_min
            if mode == "max":
                self.min_delta = 1 + self.min_delta / 100
                self.is_better = self.relative_max

    def absolute_min(self, x: float, best: float) -> bool:
        return x < best - self.min_delta

    def absolute_max(self, x: float, best: float) -> bool:
        return x > best + self.min_delta

    def relative_min(self, x: float, best: float) -> bool:
        return x < best * self.min_delta

    def relative_max(self, x: float, best: float) -> bool:
        return x > best * self.min_delta


def init_early_stopping(args) -> EarlyStopping:
    return EarlyStopping(mode="min", min_delta=0.0, patience=args.es_patience)
