import logging
import os
from datetime import datetime

from inference_config import inference_config

datetime_prefix = datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def get_logger(name, datetime_prefix=datetime_prefix):
    logging_dir = os.path.join(inference_config.log_cache, "python_logs")
    if not os.path.exists(logging_dir):
        # Create the directory if it does not exist
        os.makedirs(logging_dir)

    logger = logging.getLogger(name)
    # file_handler does logging to file
    file_handler = logging.FileHandler(
        os.path.join(logging_dir, f"{datetime_prefix}_inference_api.log")
    )
    # stream_handler does logging to stdout
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    return logger
