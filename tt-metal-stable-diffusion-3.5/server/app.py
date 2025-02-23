from flask import Flask, request, jsonify, send_from_directory
from http import HTTPStatus
import os
from server.queue import TaskQueue
from utils.authentication import api_key_required


app = Flask(__name__)

# Initialize the task queue
task_queue = TaskQueue()


# worker thread to process the task queue
def worker():
    while True:
        # get task if one exists, otherwise block
        task_id = task_queue.get_task()
        if task_id:
            task_queue.process_task(task_id)


@app.route("/")
def hello_world():
    return jsonify({"message": "OK\n"}), 200


@app.route("/enqueue", methods=["POST"])
@api_key_required
def enqueue_prompt():
    data = request.get_json()
    prompt = data.get("prompt")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), HTTPStatus.BAD_REQUEST

    # Enqueue the prompt and start processing
    task_id = task_queue.enqueue_task(prompt)
    return jsonify({"task_id": task_id, "status": "Enqueued"}), HTTPStatus.CREATED


@app.route("/status/<task_id>", methods=["GET"])
@api_key_required
def get_task_status(task_id):
    task_status = task_queue.get_task_status(task_id)
    if not task_status:
        return jsonify({"error": "Task not found"}), HTTPStatus.NOT_FOUND
    return jsonify({"task_id": task_id, "status": task_status["status"]}), HTTPStatus.OK


@app.route("/fetch_image/<task_id>", methods=["GET"])
@api_key_required
def fetch_image(task_id):
    task_status = task_queue.get_task_status(task_id)
    if not task_status:
        return jsonify({"error": "Task not found"}), HTTPStatus.NOT_FOUND

    if task_status["status"] != "Completed":
        return jsonify({"error": "Task not completed yet"}), HTTPStatus.BAD_REQUEST

    image_path = task_status["image_path"]
    directory = os.getcwd()  # get the current working directory
    return send_from_directory(directory, image_path)
