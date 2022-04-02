import logging
from logging import getLogger

formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)


def getModuleLogger(module, level=logging.INFO):
    logger = getLogger(module)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger