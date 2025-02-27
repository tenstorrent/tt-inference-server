from flask import Flask, request, Response, jsonify
from http import HTTPStatus
from server.task_queue import TaskQueue
import threading
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
        # enqeue task, when task is complete, this done_task will be called by
        # the task_queue to unblock this thread
        done_event = threading.Event()
        task_id = task_queue.enqueue_task(file, done_event)

        # wait for task to be complete
        print("WAITING")
        done_event.wait()
        print("DONE WAITING")

        # get completed task
        completed_task = task_queue.get_task_status(task_id)

        # Return the transcription result
        transcribed_output = completed_task["transcription"]
        return Response(transcribed_output, content_type="text/html; charset=utf-8")

    except Exception as e:
        return jsonify({"error": str(e)}), HTTPStatus.INTERNAL_SERVER_ERROR
