import logging
import sys
import config


def get_logger(name):
    logger = logging.getLogger(name)
    console_handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("[%(name)s @ %(levelname)s]: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.setLevel(config.LOG_LEVEL)
    return logger
