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
    handler = logging.FileHandler(
        os.path.join(logging_dir, f"{datetime_prefix}_inference_api.log")
    )
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
