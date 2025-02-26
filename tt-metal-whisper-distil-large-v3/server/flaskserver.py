from flask import Flask, request, Response, jsonify
from http import HTTPStatus
from server.task_queue import TaskQueue
from utils.authentication import api_key_required


app = Flask(__name__)

# Initialize the task queue
task_queue = TaskQueue()


# worker thread to process the task queue
def create_worker():
    while True:
        # get task if one exists, otherwise block
        task_id = task_queue.get_task()
        if task_id:
            task_queue.process_task(task_id)


@app.route("/")
def hello_world():
    return jsonify({"message": "OK\n"}), 200


@app.route("/inference", methods=["POST"])
@api_key_required
def inference():
    # check if a file was uploaded
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), HTTPStatus.BAD_REQUEST
    file = request.files["file"]

    # check if the file is valid
    if file.filename == "":
        return jsonify({"error": "No selected file"}), HTTPStatus.BAD_REQUEST

    # ensure the file is a .wav file
    if not file.filename.lower().endswith(".wav"):
        return jsonify(
            {"error": "Invalid file format, please upload a .wav file"}
        ), HTTPStatus.BAD_REQUEST

    try:
        # enqeue task
        task_id = task_queue.enqueue_task(file)

        # wait for task to be complete
        # TODO: use threading.Event() instead of polling
        while (completed_task := task_queue.get_task_status(task_id))[
            "status"
        ] != "Completed":
            pass

        # Return the transcription result
        transcribed_output = completed_task["transcription"]
        return Response(transcribed_output, content_type="text/html; charset=utf-8")

    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR
