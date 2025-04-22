import base64
from flask import Flask, request, jsonify
import functools
from http import HTTPStatus
from loguru import logger
import numpy as np
from scipy.io import wavfile
from server.model import SAMPLING_RATE
from server.task_queue import TaskQueue
import threading
import time
from utils.authentication import api_key_required


app = Flask(__name__)

# Initialize the task queue
task_queue = TaskQueue()
# Create task semaphore, used to signal only one thread when processing is finished
work_finished_semaphore = threading.Semaphore(0)


# worker thread to process the task queue
def create_worker():
    while True:
        # get task if one exists, otherwise block
        task_id = task_queue.get_task()
        if task_id:
            task_queue.process_task(task_id)
            # signal work is done
            work_finished_semaphore.release()


@app.route("/")
def hello_world():
    return jsonify({"message": "OK\n"}), 200


def return_exceptions(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Attempt to call the wrapped function
            return func(*args, **kwargs)
        except Exception as e:
            # Catch all exceptions and return an error message
            return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR

    return wrapper


@app.route("/inference", methods=["POST"])
@api_key_required
@return_exceptions
def inference():
    # A file OR JSON of form {"byte_array": array<fp32>, "sampling_rate": int}
    # must be sent
    byte_array = None
    sampling_rate = None

    def _validate_json(data):
        required_keys = ("byte_array", "sampling_rate")
        for key in required_keys:
            if key not in data:
                return jsonify(
                    {"error": f"Missing required key in JSON data: '{key}'"}
                ), HTTPStatus.BAD_REQUEST

    def _validate_file(file):
        # Check if the file is valid
        if file.filename == "":
            return jsonify({"error": "No selected file"}), HTTPStatus.BAD_REQUEST

        # Ensure the file is a .wav file
        if not file.filename.lower().endswith(".wav"):
            return jsonify(
                {"error": "Invalid file format, please upload a .wav file"}
            ), HTTPStatus.BAD_REQUEST

    def _verify_sampling_rate(sampling_rate):
        if sampling_rate != SAMPLING_RATE:
            return jsonify(
                {"error": f"sampling_rate must be 16,000, received {sampling_rate}"}
            )

    # Check if the content type is either application/json or multipart/form-data
    content_type = request.content_type

    # Case 1: If content type is application/json, check for JSON data
    if content_type == "application/json":
        # Ensure JSON data exists
        data = request.get_json()
        if data is None:
            return jsonify(
                {"error": "Request contains no JSON data"}
            ), HTTPStatus.BAD_REQUEST

        # Validate JSON contents
        if (error := _validate_json(data)) is not None:
            return error

        # Parse sampling rate
        sampling_rate = data.get("sampling_rate")

        # Decode base64 string into NumPy fp32 array
        base64_bytes = data.get("byte_array")
        byte_buffer = base64.b64decode(base64_bytes)
        byte_array = np.frombuffer(byte_buffer, dtype=np.float32)

    # Case 2: If content type is multipart/form-data, check for file
    elif content_type.startswith("multipart/form-data"):
        # Ensure a file is included in the form
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file uploaded"}), HTTPStatus.BAD_REQUEST

        # Validate file
        if (error := _validate_file(file)) is not None:
            return error

        # Read wavfile data and sampling rate
        start_time = time.time()
        sampling_rate, byte_array = wavfile.read(file)
        logger.info(f"WAV read latency: {time.time() - start_time}")

    # Case 3: If both JSON and file are sent, return an error
    elif "json" in content_type and "form-data" in content_type:
        return jsonify(
            {"error": "Cannot send both JSON data and a file"}
        ), HTTPStatus.BAD_REQUEST

    # Case 4: If neither JSON nor file is sent
    else:
        return jsonify(
            {"error": "Request must contain either JSON data or a file, not both"}
        ), HTTPStatus.BAD_REQUEST

    # Validate sampling rate
    if (error := _verify_sampling_rate(sampling_rate)) is not None:
        return error

    # Enqeue task, when task is complete, this done_task will be called by
    # the task_queue to unblock this thread
    done_event = threading.Event()
    task_id = task_queue.enqueue_task(byte_array, sampling_rate, done_event)

    # Wait for task to be complete
    done_event.wait()

    # Get completed task
    completed_task = task_queue.get_task_status(task_id)

    # Return the transcription result
    transcribed_output = completed_task["transcription"]
    return jsonify({"text": transcribed_output})
