from logging.config import dictConfig

def register():
    """
    Register the vllm_tenstorrent package.
    """
    return "tt_model_runners.vllm.platform.TTPlatform"