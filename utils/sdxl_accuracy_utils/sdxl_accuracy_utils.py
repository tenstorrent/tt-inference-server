# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import base64
import io
import os
import csv
import urllib.request
import statistics
import json
import logging
from PIL import Image
from utils.sdxl_accuracy_utils.clip_encoder import CLIPEncoder
from utils.sdxl_accuracy_utils.fid_score import calculate_fid_score

COCO_CAPTIONS_DOWNLOAD_PATH = "https://github.com/mlcommons/inference/raw/4b1d1156c23965172ae56eacdd8372f8897eb771/text_to_image/coco2014/captions/captions_source.tsv"

# Get the project root directory (assume this file is in utils/sdxl_accuracy_utils/)
CAPTIONS_PATH = "utils/sdxl_accuracy_utils/coco_data/captions.tsv"
COCO_STATISTICS_PATH = "utils/sdxl_accuracy_utils/coco_data/val2014.npz"
ACCURACY_REFERENCE_PATH = "evals/eval_targets/model_accuracy_reference.json"

logger = logging.getLogger(__name__)


def sdxl_get_prompts(
    start_from,
    num_prompts,
):
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


def decode_base64_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def save_images_as_pil(status_list: list, output_folder: str):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for idx, status in enumerate(status_list):
        # Decode and convert to PIL Image
        pil_image = decode_base64_image(status.base64image)

        # Create filename
        filename = f"image_{idx + 1}_{status.prompt.replace(' ', '_')}.png"
        image_path = os.path.join(output_folder, filename)

        # Save PIL Image
        pil_image.save(image_path)
        print(f"Image saved to {image_path}")


def calculate_metrics(status_list: list):
    prompts = [status.prompt for status in status_list]
    images = [decode_base64_image(status.base64image) for status in status_list]

    clip = CLIPEncoder()
    clip_scores = []
    for idx, image in enumerate(images):
        clip_scores.append(100 * clip.get_clip_score(prompts[idx], image).item())

    average_clip_score = sum(clip_scores) / len(clip_scores)
    deviation_clip_score = "N/A"
    fid_score = "N/A"

    deviation_clip_score = statistics.stdev(clip_scores)
    fid_score = calculate_fid_score(images, str(COCO_STATISTICS_PATH))

    print(f"FID score: {fid_score}")
    print(f"Average CLIP Score: {average_clip_score}")
    print(f"Standard Deviation of CLIP Scores: {deviation_clip_score}")

    return fid_score, average_clip_score, deviation_clip_score


def calculate_accuracy_check(fid_score, average_clip_score, num_prompts, model_name):
    logging.info(
        f"Calculating accuracy check for FID: {fid_score}, CLIP: {average_clip_score}, Prompts: {num_prompts}, Model name: {model_name}"
    )
    if num_prompts not in set([100, 5000]):
        logging.warning(
            f"⚠️ Number of prompts {num_prompts} is not supported for accuracy check."
        )
        return 0

    # Load reference data
    reference_data = _load_accuracy_reference()

    # Check if model exists in reference data
    if model_name not in reference_data:
        logging.warning(f"⚠️ Model '{model_name}' not found in accuracy reference data.")
        return 0

    # Extract the accuracy ranges for the specific model and prompt count
    accuracy_data = reference_data[model_name]["accuracy"]

    # Extract the two ranges
    fid_valid_range = accuracy_data[str(num_prompts)]["fid_valid_range"]
    clip_valid_range = accuracy_data[str(num_prompts)]["clip_valid_range"]

    # Calculate approximate ranges (±3%)
    fid_approx_range = [0.97 * fid_valid_range[0], 1.03 * fid_valid_range[1]]
    clip_approx_range = [0.97 * clip_valid_range[0], 1.03 * clip_valid_range[1]]

    # Check if scores are within approximate ranges
    fid_approx = fid_approx_range[0] <= fid_score <= fid_approx_range[1]
    clip_approx = clip_approx_range[0] <= average_clip_score <= clip_approx_range[1]

    return 2 if fid_approx and clip_approx else 3


def _load_accuracy_reference():
    """Load accuracy reference data from JSON file."""
    logging.info(f"Loading accuracy reference from: {ACCURACY_REFERENCE_PATH}")
    try:
        with open(ACCURACY_REFERENCE_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Accuracy reference file not found: {ACCURACY_REFERENCE_PATH}"
        )
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in accuracy reference file: {e}")
