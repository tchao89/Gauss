import logging


LOGGER_NAME = "logger"
LOGGER_BASIC_CONFIG = {
    "filename": "./cache/info.log",
    "filemode": "a",
    "format": "%(asctime)s %(name)s [fnm:%(filename)s lno:%(lineno)d func:%(funcName)s] %(levelname)s : %(message)s ",
    "datefmt": "%Y-%m-%d %H:%M:%S.%f",
    "level": logging.WARNING,
}
