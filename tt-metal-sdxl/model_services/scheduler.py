

import asyncio
from multiprocessing import Process, Queue

from fastapi import HTTPException
from config.settings import Settings, get_settings
from model_services.device_worker import device_worker
from utils.helpers import log_execution_time
from utils.logger import TTLogger


class Scheduler:
    @log_execution_time("Scheduler init")
    def __init__(self):
        settings = get_settings()
        self.logger = TTLogger()
        self.isReady = False
        worker_count = self._getWorkerCount(settings)
        self.worker_count = worker_count
        # a queue containting signals for warmup
        self.warmup_signals_queue = Queue(worker_count)
        self.task_queue = Queue(self._get_max_queue_size(settings))
        self.result_queue = Queue()
        self.error_queue = Queue()
        self.result_futures = {}
        self.workers = []
        self.ready_devices = []
        # init queue
        self.listener_task_ref = None
        self.device_warmup_listener_ref = None
        self.error_queue_listener_ref = None
        self.listener_running = True
        self.device_warmup_listener_running = True

    def is_queue_full(self):
        return self.task_queue.full()

    @log_execution_time("Scheduler image processing")
    def process_request(self, request):
        self.checkIsModelReady()
        if (self.task_queue.full()):
            raise HTTPException(500, "Task queue is full")
        self.task_queue.put((request))

    def checkIsModelReady(self) -> bool:
        if (self.isReady is not True):
            raise HTTPException(405, "Model is not ready")
        return True

    @log_execution_time("Workes creation")
    def startWorkers(self):
        # keep result listener in the main event loop
        self.listener_task_ref = asyncio.create_task(self.result_listener())

        # keep device warmup listener in the main event loop, it'll close soon
        self.device_warmup_listener_ref = asyncio.create_task(self.device_warmup_listener())
        # keep error listener in the main event loop
        self.error_queue_listener_ref = asyncio.create_task(self.error_listener())

        # Spawn one process per worker
        for i in range(self.worker_count):
            p = Process(target=device_worker, args=(i, self.task_queue, self.result_queue, self.warmup_signals_queue, self.error_queue))
            p.start()
            self.workers.append(p)
        self.logger.info(f"Workers started: {self.worker_count}")

    async def result_listener(self):
        while self.listener_running:
            task_id, image = await asyncio.to_thread(self.result_queue.get)
            if task_id is None:
                self.listener_running = False
                self.listener_task_ref.cancel()
                break
            future = self.result_futures.pop(task_id, None)
            if future:
                future.set_result(image)
        self.logger.info("Result listener stopped")

    async def error_listener(self):
        while self.listener_running:
            task_id, error = await asyncio.to_thread(self.error_queue.get)
            if task_id is None:
                self.listener_running = False
                self.error_queue_listener_ref.cancel()
                break
            self.logger.error(f"Error in worker {task_id}: {error}")
            # TODO add error handling for device startup error
            future = self.result_futures.pop(task_id, None)
            if future:
                future.set_exception(Exception(error))
        self.logger.info("Error listener stopped")

    async def device_warmup_listener(self):
        while self.device_warmup_listener_running == True:
            device_id = await asyncio.to_thread(self.warmup_signals_queue.get)
            self.logger.info(f"Device {device_id} is warmed up")
            self.ready_devices.append(device_id)
            # we can accept requests as soon as one device is ready
            self.isReady = True
            if len(self.ready_devices) == self.worker_count:
                self.logger.info("All devices are warmed up and ready")
                self.device_warmup_listener_running = False
                # close queue, we're done
                self._close_queues([self.warmup_signals_queue])
        self.logger.info("Device warmup listener is done")


    def stopWorkers(self):
        self.logger.info("Stopping workers")
        for worker in self.workers:
            self.task_queue.put(None)
            worker.kill()
            worker.join()
        self.workers.clear()
        self.result_queue.put((None, None))  # Stop the result listener
        self.error_queue.put((None, None))  # Stop the error listener
        if self.device_warmup_listener_ref:
            self.device_warmup_listener_ref.cancel()
        self.result_futures.clear()
        # Clean up queues
        self._close_queues([self.task_queue, self.result_queue, self.warmup_signals_queue, self.error_queue])
        self.isReady = False
        self.logger.info("Workers stopped")

    def _getWorkerCount(self, setttings: Settings) -> int:
        try:
            workerCount = len(setttings.device_ids.split(","))
            if workerCount < 1:
                self.logger.error("Worker count is 0")
                raise ValueError("Worker count must be at least 1")
            return workerCount
        except Exception as e:
            self.logger.error(f"Erros getting workers cannot: {e}")
            raise HTTPException(status_code=500, detail="Workers cannot be initialized")

    def _get_max_queue_size(self, settings: Settings) -> int:
        try:
            max_queue_size = settings.max_queue_size
            if max_queue_size < 1:
                self.logger.error("Max queue size is 0")
                raise ValueError("Max queue size must be at least 1")
            return max_queue_size
        except Exception as e:
            self.logger.error(f"Error getting max queue size: {e}")
            raise HTTPException(status_code=500, detail="Max queue size not provided in settings")

    def _close_queues(self, queues: list[Queue]):
        queues_closed = 0
        for idx, queue in enumerate(queues):
            try:
                queue.close()
                queue.join_thread()
                queues_closed += 1
            except Exception as e:
                self.logger.error(f"Error closing queue #{idx}: {e}")

        self.logger.info(f"Queues ({queues_closed}) closed successfully")