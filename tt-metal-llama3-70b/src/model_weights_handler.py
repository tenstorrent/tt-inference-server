import os
from pathlib import Path

from inference_config import inference_config
from inference_logger import get_logger

logger = get_logger(__name__)
logger.info(f"importing {__name__}")


def remove_id_prefix(s):
    ID_PREFIX = "id_"
    if s.startswith(ID_PREFIX):
        return s[len(ID_PREFIX) :]
    return s


def get_model_weights_and_tt_cache_paths():
    # defined in ModelImpl.model_container_weights_dir(), must match
    DEFAULT_TT_METAL_CACHE_DIR_NAME = "tt_metal_cache"
    DEFAULT_MODEL_WEIGHTS_DIR_NAME = "model_weights"
    model_weights_dir_path = Path(
        inference_config.cache_root, DEFAULT_MODEL_WEIGHTS_DIR_NAME
    )
    tt_cache_dir_path = Path(
        inference_config.cache_root, DEFAULT_TT_METAL_CACHE_DIR_NAME
    )
    weights_id = inference_config.model_weights_id
    weights_path = inference_config.model_weights_path
    logger.info(f"MODEL_WEIGHTS_ID:={weights_id}")
    logger.info(f"MODEL_WEIGHTS_PATH:={weights_path}")
    # available_weights = [item for item in model_weights_dir_path.iterdir()]
    if not weights_id and not weights_path:
        return default_weights_handler(
            weights_id, model_weights_dir_path, tt_cache_dir_path
        )
    elif weights_id and weights_path:
        return custome_weights_handler(weights_id, weights_path, tt_cache_dir_path)
    elif (weights_id and not weights_path) or (not weights_id and weights_path):
        err_msg = f"Must set both MODEL_WEIGHTS_ID and MODEL_WEIGHTS_PATH. weights_path=:{weights_path}, but weights_id:={weights_id}. Available_weights:={available_weights}"
        logger.error(err_msg)
        raise ValueError(err_msg)


def default_weights_handler(weights_id, model_weights_dir_path, tt_cache_dir_path):
    # default weights
    default_name = "default"
    default_weights_id = f"id_{default_name}"
    default_weights_path = model_weights_dir_path.joinpath(default_name)
    default_tt_cache_path = tt_cache_dir_path.joinpath(remove_id_prefix(default_name))
    default_weights_path.mkdir(parents=True, exist_ok=True)
    default_tt_cache_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"using default_weights_path:={default_weights_path}")
    logger.info(f"using default_tt_cache_path:={default_tt_cache_path}")
    return default_weights_path, default_tt_cache_path


def custome_weights_handler(weights_id, weights_path, tt_cache_dir_path):
    # path should be validated from backend side already
    weights_path = Path(weights_path)
    logger.info(f"using model_weights_path:={weights_path}")
    # TODO: add validation as needed
    assert weights_path.exists()
    # use same name as weights
    tt_cache_name = f"cache_{weights_path.name}"
    tt_cache_path = tt_cache_dir_path.joinpath(tt_cache_name)
    logger.info(f"using tt_cache_path:={tt_cache_path}")
    tt_cache_path.mkdir(parents=True, exist_ok=True)
    return weights_path, tt_cache_path
