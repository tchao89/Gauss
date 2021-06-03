import logging


LOGGER_NAME = "logger"
LOGGER_BASIC_CONFIG = {
    "filename": "info.log",
    "filemode": "a",
    "format": "%(asctime)s %(name)s [pid:%(process)d pnm:%(processName)s] [tid:%(thread)d tnm:%(thread)s] [fnm:%(filename)s lno:%(lineno)d func:%(funcName)s] %(levelname)s : %(message)s ",
    "datefmt": "%Y-%m-%d %H:%M:%S.%f",
    "level": logging.INFO,
}