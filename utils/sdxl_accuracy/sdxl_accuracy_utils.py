import os
import csv
import urllib.request
import statistics
import json
from models.experimental.stable_diffusion_xl_base.utils.clip_encoder import CLIPEncoder
from models.experimental.stable_diffusion_xl_base.utils.fid_score import calculate_fid_score
from models.experimental.stable_diffusion_xl_base.utils.clip_fid_ranges import (
    accuracy_check_clip,
    accuracy_check_fid,
    get_appr_delta_metric,
    targets,
)


COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"
CAPTIONS_PATH = "models/experimental/stable_diffusion_xl_base/coco_data/captions.tsv"
OUTPUT_FOLDER = "output"
OUT_ROOT, RESULTS_FILE_NAME = "test_reports", "sdxl_test_results.json"
COCO_STATISTICS_PATH = "models/experimental/stable_diffusion_xl_base/coco_data/val2014.npz"
MODEL_NAME = "stable-diffusion-xl-base-1.0-provisional"
NUM_INFERENCE_STEPS = 20
NEGATIVE_PROMPT = "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
GUIDANCE_SCALE = 8


def sdxl_get_prompts(
    start_from,
    num_prompts,
):
    assert (
        0 <= start_from < 5000 and start_from + num_prompts <= 5000
    ), "start_from must be between 0 and 4999, and start_from + num_prompts must not exceed 5000."

    prompts = []

    if not os.path.isfile(CAPTIONS_PATH):
        os.makedirs(os.path.dirname(CAPTIONS_PATH), exist_ok=True)
        urllib.request.urlretrieve(COCO_CAPTIONS_DOWNLOAD_PATH, CAPTIONS_PATH)

    with open(CAPTIONS_PATH, "r") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t")
        next(reader)
        for index, row in enumerate(reader):
            if index < start_from:
                continue
            if index >= start_from + num_prompts:
                break
            prompts.append(row[2])

    return prompts


def save_image(image_data, name):
    image_path = os.path.join(OUTPUT_FOLDER, f'generated_image_{name}.png')
    with open(image_path, 'wb') as f:
        f.write(image_data)
    print(f"Image saved to {image_path}")
    
    
def calculate_metrics(num_prompts, images, prompts):
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
    return fid_score, average_clip_score, deviation_clip_score


def create_test_results_json(
    device,
    num_prompts,
    avg_generation_time,
    requests_times_list,   
    fid_score, 
    average_clip_score, 
    deviation_clip_score
):
    data = {
        "model": MODEL_NAME,
        "metadata": {
            "model_name": MODEL_NAME,
            "device": device,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "start_from": 0,
            "num_prompts": num_prompts,
            "negative_prompt": NEGATIVE_PROMPT,
            "guidance_scale": GUIDANCE_SCALE,
        },
        "benchmarks_summary": [
            {
                "model": MODEL_NAME,
                "device": device,
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
                "model": MODEL_NAME,
                "device": device,
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
    return data

def save_json(data):
    os.makedirs(OUT_ROOT, exist_ok=True)

    with open(
        f"{OUT_ROOT}/{RESULTS_FILE_NAME}", "w"
    ) as f:
        json.dump(data, f, indent=4)

    print(f"Test results saved to {OUT_ROOT}/{RESULTS_FILE_NAME}")