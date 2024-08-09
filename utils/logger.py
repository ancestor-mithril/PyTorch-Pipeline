import logging
import os
import sys
from typing import Optional

logger: Optional["Logger"] = None


class Logger:
    def __init__(self, logdir: str, verbose: bool, stderr: bool):
        file_logger = logging.getLogger("Pipeline.file")
        file_logger.setLevel(logging.INFO)
        os.makedirs(logdir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(logdir, 'train.log'), mode='w')
        file_logger.addHandler(file_handler)
        self.file_logger = file_logger

        console_logger = logging.getLogger("Pipeline.console")
        console_logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler(sys.stderr if stderr else sys.stdout)
        console_logger.addHandler(stream_handler)
        self.console_logger = console_logger
        self.verbose = verbose

    def log(self, *args, to_console: bool = True):
        string = ' '.join(map(str, args))
        self.file_logger.info(string)
        if self.verbose and to_console:
            self.console_logger.info(string)

    def log_both(self, *args):
        string = ' '.join(map(str, args))
        self.file_logger.info(string)
        self.console_logger.info(string)


def init_logger(logdir: str, verbose: bool, stderr: bool) -> Logger:
    global logger
    logger = Logger(logdir, verbose, stderr)
    return logger


def get_logger() -> Logger:
    global logger
    if logger is None:
        raise RuntimeError("Logger must be initialized")
    return logger
