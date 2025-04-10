from loguru import logger
import os
import random
import string
import ttnn
from models.experimental.stable_diffusion3.tt import TtStableDiffusion3Pipeline


# Global variable for the Stable Diffusion model pipeline
model_pipeline = None


# Warmup the model on app startup
def warmup_model():
    # create device, these constants are specific to n150 & n300
    device_id = 0
    device_params = {"l1_small_size": 8192, "trace_region_size": 15210496}
    dispatch_core_type = ttnn.device.DispatchCoreType.WORKER
    if ("WH_ARCH_YAML" in os.environ) and os.environ["WH_ARCH_YAML"] == "wormhole_b0_80_arch_eth_dispatch.yaml":
        dispatch_core_type = ttnn.device.DispatchCoreType.ETH
    dispatch_core_axis = ttnn.DispatchCoreAxis.ROW
    dispatch_core_config = ttnn.DispatchCoreConfig(
        dispatch_core_type, dispatch_core_axis
    )
    device_params["dispatch_core_config"] = dispatch_core_config
    device = ttnn.CreateDevice(device_id=device_id, **device_params)
    device.enable_program_cache()

    # create model pipeline
    global model_pipeline
    model_pipeline = TtStableDiffusion3Pipeline(
        checkpoint="stabilityai/stable-diffusion-3.5-medium",
        device=device,
        enable_t5_text_encoder=True,
    )

    model_pipeline.prepare(
        batch_size=1,
        width=512,
        height=512,
    )
    prompt = (
        "An epic, high-definition cinematic shot of a rustic snowy cabin glowing "
        "warmly at dusk, nestled in a serene winter landscape. Surrounded by gentle "
        "snow-covered pines and delicate falling snowflakes - captured in a rich, "
        "atmospheric, wide-angle scene with deep cinematic depth and warmth."
    )
    negative_prompt = ""

    model_pipeline(
        prompt_1=[prompt],
        prompt_2=[prompt],
        prompt_3=[prompt],
        negative_prompt_1=[negative_prompt],
        negative_prompt_2=[negative_prompt],
        negative_prompt_3=[negative_prompt],
        num_inference_steps=40,
        seed=0,
    )
    logger.info("Loading Stable Diffusion model...")
    logger.info("Model loaded and ready!")


# Function to generate an image from the given prompt
def generate_image_from_prompt(prompt):
    # Generate a random file name for the image
    random_filename = (
        "".join(random.choices(string.ascii_lowercase + string.digits, k=10)) + ".png"
    )
    image_path = os.path.join("generated_images", random_filename)

    # Generate the image using Stable Diffusion
    negative_prompt = ""
    images = model_pipeline(
        prompt_1=[prompt],
        prompt_2=[prompt],
        prompt_3=[prompt],
        negative_prompt_1=[negative_prompt],
        negative_prompt_2=[negative_prompt],
        negative_prompt_3=[negative_prompt],
        num_inference_steps=40,
        seed=None,
    )
    images[0].save(image_path)

    return image_path
