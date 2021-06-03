from datetime import datetime
import sys
import time
import logging
from logging import LogRecord
from logging.handlers import WatchedFileHandler
from typing import Optional
from utils.logger_conf import LOGGER_NAME, LOGGER_BASIC_CONFIG


class Logger:
    def __init__(self, basic_config=None):
        self.basic_config = self.__generate_basic_config()

        if basic_config is not None:
            self.basic_config.update(basic_config)

        self.logger = self.__build_logger()

    @classmethod
    def __generate_basic_config(cls):
        basic_config = LOGGER_BASIC_CONFIG

        return basic_config

    def __build_logger(self):
        default_logger = logging.getLogger(LOGGER_NAME)
        default_logger.setLevel(self.basic_config["level"])

        file_handler = WatchedFileHandler(self.basic_config["filename"], self.basic_config["filemode"], encoding="utf-8")
        file_handler.setLevel(self.basic_config["level"])

        steam_handler = logging.StreamHandler(sys.stderr)
        steam_handler.setLevel(self.basic_config["level"])

        formatter = Formatter(self.basic_config["format"], self.basic_config["datefmt"])
        file_handler.setFormatter(formatter)
        steam_handler.setFormatter(formatter)

        default_logger.addHandler(file_handler)
        default_logger.addHandler(steam_handler)

        return default_logger


class Formatter(logging.Formatter):
    def formatTime(self, record: LogRecord, datefmt: Optional[str] = ...) -> str:
        ct = self.converter(record.created)
        if datefmt:
            s = datetime.now().strftime(datefmt)
        else:
            t = time.strftime(self.default_time_format, ct)
            s = self.default_msec_format % (t, record.msecs)
        return s


logger = Logger().logger
