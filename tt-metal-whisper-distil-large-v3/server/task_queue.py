from loguru import logger
import queue
import uuid
from server.model import perform_asr


class TaskQueue:
    def __init__(self):
        self.tasks = {}
        self.task_queue = queue.Queue()

    def enqueue_task(self, audio_file, thread_event):
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = {
            "file": audio_file,
            "thread_event": thread_event,
            "status": "Pending",
            "transcription": None,
        }

        # Add the task to the queue for processing
        self.task_queue.put(task_id)
        return task_id

    def get_task(self):
        # Get the next task from the queue (this will block until there's a task)
        return self.task_queue.get()

    def process_task(self, task_id):
        task = self.tasks[task_id]
        task["status"] = "In Progress"

        # perform ASR
        logger.info(f"Processing task {task_id} for task: {task_id}")
        transcribed_output = perform_asr(task["file"])

        # update task status and store transcription
        task["status"] = "Completed"
        task["transcription"] = transcribed_output

        logger.info(f"Task {task_id} completed")

        # signal thread event complete
        task["thread_event"].set()

    def get_task_status(self, task_id):
        return self.tasks.get(task_id)
