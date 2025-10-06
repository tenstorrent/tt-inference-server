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
import argparse


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

def parse_args():
    parser = argparse.ArgumentParser(description="SDXL Accuracy Testing")
    parser.add_argument(
        "--n-prompts", 
        type=int, 
        default=2, 
        help="Number of prompts to test (default: 2)"
    )
    parser.add_argument(
        "--service-port", 
        type=int, 
        default=8000, 
        help="Port where the image generation service is running (default: 8000)"
    )
    parser.add_argument(
        "--ci-env", 
        type=int, 
        default=1, 
        help="Whether the script is running in a CI environment (default: 1)"
    )
    
    args = parser.parse_args()
    if not (2 <= args.n_prompts <= 5000):
        parser.error("--n-prompts must be between 2 and 5000")
    
    if args.ci_env not in [0, 1]:
        parser.error("--ci-env must be 0 or 1") 

    print(f"Number of prompts: {args.n_prompts}, CI Environment: {args.ci_env}, Service Port: {args.service_port}")
    return args


OUTPUT_FOLDER = "output"
CAPTIONS_PATH = "models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"
COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
COCO_STATISTICS_PATH = "models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"
NUM_INFERENCE_STEPS = 20
NEGATIVE_PROMPT = "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
GUIDANCE_SCALE = 8
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "sdxl_test_results.json"


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
        os.makedirs(os.path.dirname(captions_path), exist_ok=True)
        urllib.request.urlretrieve(COCO_CAPTIONS_DOWNLOAD_PATH, captions_path)

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

def request_one_image_base64(prompt, service_port):
    start_time = time.time()
    response = requests.post(
        f'http://127.0.0.1:{service_port}/image/generations',
        headers={
            'accept': 'application/json',
            'Authorization': 'Bearer your-secret-key',
            'Content-Type': 'application/json'
        },
        json={
            "prompt": prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "seed": 0,
            "guidance_scale": GUIDANCE_SCALE,
            "number_of_images": 1
        }
    )
    
    request_response_time = time.time() - start_time
    
    response_data = response.json()
    return response_data.get("images", [])[0], request_response_time

def generate_accuracy_images(prompts, ci_run, service_port):
    decoded_images = []
    requests_times_list = []
    for i, prompt in enumerate(prompts):
        image_base64, request_response_time = request_one_image_base64(prompt, service_port)
        requests_times_list.append(request_response_time)
        
        image_bytes = base64.b64decode(image_base64)        
        if not ci_run: save_image(image_bytes, i+1)

        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        decoded_images.append(pil_image)
        print("Generated image for prompt:", prompt)

    return decoded_images, requests_times_list
    
def save_image(image_data, name):
    image_path = os.path.join(OUTPUT_FOLDER, f'generated_image_{name}.png')
    with open(image_path, 'wb') as f:
        f.write(image_data)
    print(f"Image saved to {image_path}")

if __name__ == "__main__":
    args = parse_args()
    num_prompts = args.n_prompts
    ci_run_bool = args.ci_env == 1
    service_port = args.service_port
    
    client = ImageClient(
        all_params=None,
        model_spec=None,
        device=None,
        output_path=None,
        service_port=service_port
    )
    
    print("Checking server health...")
    is_healthy, runner_type = client.get_health()
    
    if not is_healthy:
        print("❌ Server is not healthy. Exiting...")
        exit(1)
    print(f"✅ Server is healthy! Using runner: {runner_type}")
    
    prompts = sdxl_get_prompts(CAPTIONS_PATH, start_from=0, num_prompts=num_prompts)
    if not ci_run_bool: print(prompts)

    if not ci_run_bool:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    images, requests_times_list = generate_accuracy_images(prompts, ci_run_bool, service_port)
    avg_generation_time = sum(requests_times_list) / len(requests_times_list)
    
    clip = CLIPEncoder()

    clip_scores = []

    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())

    average_clip_score = sum(clip_scores) / len(clip_scores)

    deviation_clip_score = "N/A"
    fid_score = "N/A"

    if num_prompts >= 2:
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
            "device": "N150",
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "start_from": 0,
            "num_prompts": num_prompts,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": GUIDANCE_SCALE,
        },
        "benchmarks_summary": [
            {
                "model": "sdxl",
                "device": "N150",
                "avg_gen_time": avg_generation_time,
                "target_checks": {
                    "functional": {
                        "avg_gen_time": targets["perf"]["functional"],
                        "avg_gen_time_check": 2 if targets["perf"]["functional"] >= avg_generation_time else 3,
                    },
                    "complete": {
                        "avg_gen_time": targets["perf"]["complete"],
                        "avg_gen_time_check": 2 if targets["perf"]["complete"] >= avg_generation_time else 3,
                    },
                    "target": {
                        "avg_gen_time": targets["perf"]["target"],
                        "avg_gen_time_check": 2 if targets["perf"]["target"] >= avg_generation_time else 3,
                    },
                },
                "min_gen_time": min(requests_times_list),
                "max_gen_time": max(requests_times_list),
            }
        ],
        "evals": [
            {
                "model": "sdxl",
                "device": "N150",
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
    ) as f:
        json.dump(data, f, indent=4)

    print(f"Test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")
