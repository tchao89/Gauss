import logging


LOGGER_NAME = "logger"
LOGGER_BASIC_CONFIG = {
    "filename": "/home/liangqian/PycharmProjects/Gauss/cache/info.log",
    "filemode": "a",
    "format": "[time: %(asctime)s] | [fnm:%(filename)s] | [lno:%(lineno)d] | [func:%(funcName)s] | [%(levelname)s : %(message)s] ",
    "datefmt": "%Y-%m-%d %H:%M:%S",
    "level": logging.INFO,
}
