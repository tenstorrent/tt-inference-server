import requests
import os
import sys
import base64
import time
import json
import csv
import urllib.request
from PIL import Image
import io
import time

added_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, added_path)

tt_metal_path = os.path.join(os.path.dirname(added_path), "tt-metal")
sys.path.insert(0, tt_metal_path)

original_cwd = os.getcwd()
os.chdir(tt_metal_path)

from utils.image_client import ImageClient
from models.experimental.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder
import statistics
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score
from models.experimental.stable_diffusion_xl_base.utils.clip_fid_ranges import (
    accuracy_check_clip,
    accuracy_check_fid,
    get_appr_delta_metric,
    targets,
)


SERVICE_PORT = 8000
OUTPUT_FOLDER = "output"
CAPTIONS_PATH = "models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"
COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
COCO_STATISTICS_PATH = "models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"
N_PROMPTS = 2
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "sdxl_test_results.json"
os.chdir(original_cwd)

client = ImageClient(
    all_params=None,
    model_spec=None, 
    device=None,
    output_path=None,
    service_port=SERVICE_PORT
)

def sdxl_get_prompts(
    captions_path,
    start_from,
    num_prompts,
):
    assert (
        0 <= start_from < 5000 and start_from + num_prompts <= 5000
    ), "start_from must be between 0 and 4999, and start_from + num_prompts must not exceed 5000."

    prompts = []

    if not os.path.isfile(captions_path):
        # logger.info(f"File {captions_path} not found. Downloading...")
        os.makedirs(os.path.dirname(captions_path), exist_ok=True)
        urllib.request.urlretrieve(COCO_CAPTIONS_DOWNLOAD_PATH, captions_path)
        # logger.info("Download complete.")

    with open(captions_path, "r") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)
        for index, row in enumerate(reader):
            if index < start_from:
                continue
            if index >= start_from + num_prompts:
                break
            prompts.append(row[2])

    return prompts

def request_images_base64(prompt, number_of_images):
    response = requests.post(
        f'http://127.0.0.1:{SERVICE_PORT}/image/generations',
        headers={
            'accept': 'application/json',
            'Authorization': 'Bearer your-secret-key',
            'Content-Type': 'application/json'
        },
        json={
            "prompt": prompt,
            "negative_prompt": "normal quality, low quality, worst quality, low res, blurry, nsfw, nude",
            "num_inference_steps": 20,
            "seed": 0,
            "guidance_scale": 8,
            "number_of_images": number_of_images
        }
    )
    
    
    response_data = response.json()
    return response_data.get("images", [])

def generate_accuracy_images(prompts):
    decoded_images = []
    total_generation_time = 0.0
    for i, prompt in enumerate(prompts):
        start_time = time.time()
        image_base64 = request_images_base64(prompt, 1)[0]
        end_time = time.time()
        total_generation_time += (end_time - start_time)
        
        image_bytes = base64.b64decode(image_base64)        
        save_image(image_bytes, i)

        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        decoded_images.append(pil_image)
        print("Generated image for prompt:", prompt)
        
        # decoded_images.append(base64.b64decode(image_base64[0]))
    
    avg_generation_time = total_generation_time / len(prompts)
    return decoded_images, avg_generation_time
    
def save_image(image_data, name):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    image_path = os.path.join(OUTPUT_FOLDER, f'generated_image_{name}.png')
    with open(image_path, 'wb') as f:
        f.write(image_data)
    print(f"Image saved to {image_path}")

if __name__ == "__main__":
    print("Checking server health...")
    is_healthy, runner_type = client.get_health()
    num_prompts = N_PROMPTS
    
    if not is_healthy:
        print("❌ Server is not healthy. Exiting...")
        exit(1)
    print(f"✅ Server is healthy! Using runner: {runner_type}")
    
    prompts = sdxl_get_prompts(CAPTIONS_PATH, start_from=0, num_prompts=N_PROMPTS)
    print(prompts)

    images, avg_generation_time = generate_accuracy_images(prompts)
    
    clip = CLIPEncoder()

    clip_scores = []

    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())

    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"

    if N_PROMPTS >= 2:
        deviation_clip_score = statistics.stdev(clip_scores)
        fid_score = calculate_fid_score(images, COCO_STATISTICS_PATH)
    else:
        print("FID score is not calculated for less than 2 prompts.")

    print(f"FID score: {fid_score}")
    print(f"Average CLIP Score: {average_clip_score}")
    print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")
    
    
    
    data = {
        "model": "sdxl",
        "metadata": {
            "model_name": "sdxl",
            # "device": get_device_name(),
            # "device_vae": vae_on_device,
            # "capture_trace": capture_trace,
            # "encoders_on_device": encoders_on_device,
            # "num_inference_steps": num_inference_steps,
            "start_from": 0,
            "num_prompts": N_PROMPTS,
            # "negative_prompt": negative_prompt,
            # "guidance_scale": guidance_scale,
        },
        "benchmarks_summary": [
            {
                "model": "sdxl",
                # "device": get_device_name(),
                "avg_gen_time": avg_generation_time,
                "target_checks": {
                    "functional": {
                        "avg_gen_time": targets["perf"]["functional"],
                        "avg_gen_time_check": 3 if targets["perf"]["functional"] >= avg_generation_time else 2,
                    },
                    "complete": {
                        "avg_gen_time": targets["perf"]["complete"],
                        "avg_gen_time_check": 3 if targets["perf"]["complete"] >= avg_generation_time else 2,
                    },
                    "target": {
                        "avg_gen_time": targets["perf"]["target"],
                        "avg_gen_time_check": 3 if targets["perf"]["target"] >= avg_generation_time else 2,
                    },
                },
                # "average_denoising_time": profiler.get("denoising_loop"),
                # "average_vae_time": profiler.get("vae_decode"),
                # "min_gen_time": min(profiler.times["end_to_end_generation"]),
                # "max_gen_time": max(profiler.times["end_to_end_generation"]),
                # "average_encoding_time": profiler.get("encode_prompts"),
            }
        ],
        "evals": [
            {
                "model": "sdxl",
                # "device": get_device_name(),
                "average_clip": average_clip_score,
                "deviation_clip": deviation_clip_score,
                "approx_clip_accuracy_check": accuracy_check_clip(average_clip_score, num_prompts, mode="approx"),
                "average_clip_accuracy_check": accuracy_check_clip(average_clip_score, num_prompts, mode="valid"),
                "delta_clip": get_appr_delta_metric(average_clip_score, num_prompts, score_type="clip"),
                "fid_score": fid_score,
                "approx_fid_accuracy_check": accuracy_check_fid(fid_score, num_prompts, mode="approx"),
                "fid_score_accuracy_check": accuracy_check_fid(fid_score, num_prompts, mode="valid"),
                "delta_fid": get_appr_delta_metric(fid_score, num_prompts, score_type="fid"),
                "accuracy_check_approx": min(
                    accuracy_check_fid(fid_score, num_prompts, mode="approx"),
                    accuracy_check_clip(average_clip_score, num_prompts, mode="approx"),
                ),
                "accuracy_check_delta": min(
                    accuracy_check_fid(fid_score, num_prompts, mode="delta"),
                    accuracy_check_clip(average_clip_score, num_prompts, mode="delta"),
                ),
                "accuracy_check_valid": min(
                    accuracy_check_fid(fid_score, num_prompts, mode="valid"),
                    accuracy_check_clip(average_clip_score, num_prompts, mode="valid"),
                ),
            }
        ],
    }

    os.makedirs(OUT_ROOT, exist_ok=True)

    with open(
        f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w"
    ) as f:  # this is for CI and test_sdxl_accuracy_with_reset.py compatibility
        json.dump(data, f, indent=4)

    print(f"Test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")