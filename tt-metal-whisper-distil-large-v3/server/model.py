from loguru import logger
from scipy.io import wavfile
import time
import ttnn
from models.experimental.functional_whisper.demo.demo import (
    create_functional_whisper_for_conditional_generation_inference_pipeline,
)
from models.experimental.functional_whisper.tt import (
    ttnn_optimized_functional_whisper as ttnn_model,
)


# Global variable for the Whisper model pipeline
model_pipeline = None


# Warmup the model on app startup
def warmup_model():
    # create device, these constants are specific to n150 & n300
    device_id = 0
    device_params = {}
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
    dispatch_core_config = ttnn.DispatchCoreConfig(
        dispatch_core_type, dispatch_core_axis
    )
    device_params["dispatch_core_config"] = dispatch_core_config
    device = ttnn.CreateDevice(device_id=device_id, **device_params)
    device.enable_program_cache()
    device.enable_async(True)

    # create model pipeline
    global model_pipeline
    model_pipeline = (
        create_functional_whisper_for_conditional_generation_inference_pipeline(
            ttnn_model,
            device,
        )
    )

    # warmup model pipeline
    input_file_path = "/home/container_app_user/app/server/17646385371758249908.wav"
    sampling_rate, data = wavfile.read(input_file_path)
    _ttnn_output = model_pipeline(data, sampling_rate, stream=False)

    logger.info("Loading Stable Diffusion model...")
    logger.info("Model loaded and ready!")


# Function to perform asr on an audio file, the output decoded tokens are returned
def perform_asr(audio_file):
    start_time = time.time()
    sampling_rate, data = wavfile.read(audio_file)
    logger.info(f"WAV read latency: {time.time() - start_time}")
    ttnn_output = model_pipeline(data, sampling_rate, stream=False)
    logger.info(f"Model Pipeline latency: {time.time() - start_time}")

    return ttnn_output
