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
from utils.sdxl_accuracy.sdxl_accuracy_utils import (
    sdxl_get_prompts,
    save_image,
    calculate_metrics,
    create_test_results_json,
    save_json,
    OUTPUT_FOLDER,
    NEGATIVE_PROMPT,
    NUM_INFERENCE_STEPS,
    GUIDANCE_SCALE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="SDXL Accuracy Testing")
    parser.add_argument(
        "--n-prompts", 
        type=int, 
        default=100, 
        help="Number of prompts to test (default: 100)"
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
    parser.add_argument(
        "--device",
        type=str,
        default="n150",
        help="Device type (default: n150)",
    )
    
    args = parser.parse_args()
    if not (2 <= args.n_prompts <= 5000):
        parser.error("--n-prompts must be between 2 and 5000")
    
    if args.ci_env not in [0, 1]:
        parser.error("--ci-env must be 0 or 1") 

    print(f"Number of prompts: {args.n_prompts}, CI Environment: {args.ci_env}, Service Port: {args.service_port}, Device: {args.device}")
    return args


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


if __name__ == "__main__":
    args = parse_args()
    num_prompts = args.n_prompts
    ci_run_bool = args.ci_env == 1
    service_port = args.service_port
    device = args.device.lower()
    
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
    
    prompts = sdxl_get_prompts(start_from=0, num_prompts=num_prompts)
    if not ci_run_bool: print(prompts)

    if not ci_run_bool:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    images, requests_times_list = generate_accuracy_images(prompts, ci_run_bool, service_port)
    avg_generation_time = sum(requests_times_list) / len(requests_times_list)

    fid_score, average_clip_score, deviation_clip_score = calculate_metrics(num_prompts, images, prompts)

    data = create_test_results_json(
        device,
        num_prompts,
        avg_generation_time,
        requests_times_list,   
        fid_score, 
        average_clip_score, 
        deviation_clip_score
    )
    
    save_json(data)

    
