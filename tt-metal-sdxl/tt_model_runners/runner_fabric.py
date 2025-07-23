
from config.settings import settings
from tt_model_runners.base_device_runner import DeviceRunner


def get_device_runner() -> DeviceRunner:
        model_runner = settings.model_runner
        if model_runner == "tt-sdxl":
            from tt_model_runners.sdxl_runner import TTSDXLRunner
            return TTSDXLRunner()
        else:
            raise ValueError(f"Unsupported model runner: {model_runner}")