import logging
import os
import sys

logger = None


class Logger:
    def __init__(self, logdir: str, verbose: bool):
        file_logger = logging.getLogger("Pipeline.file")
        file_logger.setLevel(logging.INFO)
        os.makedirs(logdir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(logdir, 'train.log'), mode='w')
        file_logger.addHandler(file_handler)
        self.file_logger = file_logger

        console_logger = logging.getLogger("Pipeline.console")
        console_logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler(sys.stdout)
        console_logger.addHandler(stream_handler)
        self.console_logger = console_logger
        self.verbose = verbose

    def log(self, *args, to_console: bool = True, force_console: bool = False):
        string = ' '.join(map(str, args))
        self.file_logger.info(string)
        if self.verbose and to_console or force_console:
            self.console_logger.info(string)


def init_logger(logdir: str, verbose: bool) -> Logger:
    global logger
    logger = Logger(logdir, verbose)
    return logger


def get_logger() -> logger:
    global logger
    if logger is None:
        raise RuntimeError("Logger must be initialized")
    return logger
