import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter(
        "%(levelname)-5s | %(name)s | %(message)s",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
