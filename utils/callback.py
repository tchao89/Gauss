from typing import List
from utils.Logger import logger


def multiprocess_callback(subprocess_result):
    logger.info("flags: " + str(subprocess_result))
