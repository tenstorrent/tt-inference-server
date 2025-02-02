# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from flask import (
    abort,
    Flask,
    request,
    jsonify,
    send_from_directory,
)
import json
import logging
import os
import atexit
import time
import threading
from http import HTTPStatus
from utils.authentication import api_key_required

import subprocess
import signal
import sys


# initialize logger
logger = logging.getLogger(__name__)


# script to run in background
script = "pytest models/demos/wormhole/stable_diffusion/demo/web_demo/sdserver.py"

# Start script using subprocess
process1 = subprocess.Popen(script, shell=True)


# Function to terminate both processes and kill port 5000
def signal_handler(sig, frame):
    logger.info("Terminating processes...")
    process1.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

app = Flask(__name__)

# var to indicate ready state
ready = False

# internal json prompt file
json_file_path = (
    "models/demos/wormhole/stable_diffusion/demo/web_demo/input_prompts.json"
)


@app.route("/")
def hello_world():
    return "Hello, World!"


def submit_prompt(prompt_file, prompt):
    if not os.path.isfile(prompt_file):
        with open(prompt_file, "w") as f:
            json.dump({"prompts": []}, f)

    with open(prompt_file, "r") as f:
        prompts_data = json.load(f)

    prompts_data["prompts"].append({"prompt": prompt, "status": "not generated"})

    with open(prompt_file, "w") as f:
        json.dump(prompts_data, f, indent=4)


def warmup():
    sample_prompt = "Unicorn on a banana"
    # submit sample prompt to perform tracing and server warmup
    submit_prompt(json_file_path, sample_prompt)
    global ready
    while not ready:
        with open(json_file_path, "r") as f:
            prompts_data = json.load(f)
        # sample prompt should be first prompt
        sample_prompt_data = prompts_data["prompts"][0]
        if sample_prompt_data["prompt"] == sample_prompt:
            # TODO: remove this and replace with status check == "done"
            # to flip ready flag
            if sample_prompt_data["status"] == "done":
                ready = True
                logger.info("Warmup complete")
        time.sleep(3)


# start warmup routine in background while server starts
warmup_thread = threading.Thread(target=warmup, name="warmup")
warmup_thread.start()


@app.route("/health")
def health_check():
    if not ready:
        abort(HTTPStatus.SERVICE_UNAVAILABLE, description="Server is not ready yet")
    return jsonify({"message": "OK\n"}), 200


@app.route("/submit", methods=["POST"])
@api_key_required
def submit():
    global ready
    if not ready:
        abort(HTTPStatus.SERVICE_UNAVAILABLE, description="Server is not ready yet")
    data = request.get_json()
    prompt = data.get("prompt")
    logger.info(f"Prompt: {prompt}")

    submit_prompt(json_file_path, prompt)

    return jsonify({"message": "Prompt received and added to queue."})


@app.route("/update_status", methods=["POST"])
def update_status():
    data = request.get_json()
    prompt = data.get("prompt")

    with open(json_file_path, "r") as f:
        prompts_data = json.load(f)

    for p in prompts_data["prompts"]:
        if p["prompt"] == prompt:
            p["status"] = "generated"
            break

    with open(json_file_path, "w") as f:
        json.dump(prompts_data, f, indent=4)

    return jsonify({"message": "Prompt status updated to generated."})


@app.route("/get_image", methods=["GET"])
def get_image():
    image_name = "interactive_512x512_ttnn.png"
    directory = os.getcwd()  # Get the current working directory
    return send_from_directory(directory, image_name)


@app.route("/image_exists", methods=["GET"])
def image_exists():
    image_path = "interactive_512x512_ttnn.png"
    if os.path.isfile(image_path):
        return jsonify({"exists": True}), 200
    else:
        return jsonify({"exists": False}), 200


@app.route("/clean_up", methods=["POST"])
def clean_up():
    with open(json_file_path, "r") as f:
        prompts_data = json.load(f)

    prompts_data["prompts"] = [
        p for p in prompts_data["prompts"] if p["status"] != "done"
    ]

    with open(json_file_path, "w") as f:
        json.dump(prompts_data, f, indent=4)

    return jsonify({"message": "Cleaned up done prompts."})


@app.route("/get_latest_time", methods=["GET"])
def get_latest_time():
    if not os.path.isfile(json_file_path):
        return jsonify({"message": "No prompts found"}), 404

    with open(json_file_path, "r") as f:
        prompts_data = json.load(f)

    # Filter prompts that have a total_acc time available
    completed_prompts = [p for p in prompts_data["prompts"] if "total_acc" in p]

    if not completed_prompts:
        return jsonify({"message": "No completed prompts with time available"}), 404

    # Get the latest prompt with total_acc
    latest_prompt = completed_prompts[-1]  # Assuming prompts are in chronological order

    return (
        jsonify(
            {
                "prompt": latest_prompt["prompt"],
                "total_acc": latest_prompt["total_acc"],
                "batch_size": latest_prompt["batch_size"],
                "steps": latest_prompt["steps"],
            }
        ),
        200,
    )


def cleanup():
    if os.path.isfile(
        "models/demos/wormhole/stable_diffusion/demo/web_demo/input_prompts.json"
    ):
        os.remove(
            "models/demos/wormhole/stable_diffusion/demo/web_demo/input_prompts.json"
        )
        logger.info("Deleted json")

    if os.path.isfile("interactive_512x512_ttnn.png"):
        os.remove("interactive_512x512_ttnn.png")
        logger.info("Deleted image")

    signal_handler(None, None)
    logger.info("Cleanup complete")


atexit.register(cleanup)


def create_server():
    return app
