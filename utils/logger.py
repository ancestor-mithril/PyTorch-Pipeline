import logging
import os
import sys


def init_logger(logdir: str, verbose: bool):
    logger = logging.getLogger("Pipeline")
    logger.setLevel(logging.INFO)
    os.makedirs(logdir, exist_ok=True)
    fileHandler = logging.FileHandler(os.path.join(logdir, 'train.log'), mode='w')
    logger.addHandler(fileHandler)
    if verbose:
        streamHandler = logging.StreamHandler(sys.stdout)
        logger.addHandler(streamHandler)
    return logger
