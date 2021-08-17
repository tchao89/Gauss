import sys
from loguru import logger


logger.add("/home/liangqian/PycharmProjects/Gauss/test/file.log", format="{time :YYYY-MM-DD at HH:mm:ss } {level} | {level} | {message}", filter="my_module", level="INFO")
logger.info("test code...")

# For scripts
config = {
    "handlers": [
        {"sink": sys.stdout, "format": "{time} - {message}"},
        {"sink": "file.log", "serialize": True},
    ],
    "extra": {"user": "someone"}
}
