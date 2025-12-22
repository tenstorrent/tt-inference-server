# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import time
import uuid
from multiprocessing import Process, Queue
from threading import Lock

from config.settings import settings
from model_services.device_worker import setup_cpu_threading_limits
from model_services.tt_queue import TTQueue
from utils.logger import TTLogger


def _process_worker_tasks(
    task_queue,
    result_queue,
    error_queue,
    worker_name,
    worker_id,
    worker_function,
    worker_context_setup=None,
):
    """Worker process - similar to device_worker"""
    logger = TTLogger()
    logger.info(f"{worker_name} worker {worker_id} started")

    setup_cpu_threading_limits("2")

    worker_context = None
    if worker_context_setup:
        try:
            worker_context = worker_context_setup()
            logger.info(f"{worker_name} worker {worker_id} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {worker_name} worker {worker_id}: {e}")
            return

    while True:
        try:
            task_data = task_queue.get()

            if task_data is None:  # Shutdown signal
                logger.info(f"{worker_name} worker received shutdown signal")
                break

            task_id = task_data[0]
            logger.info(f"Processing {worker_name}-{worker_id} task {task_id}")
            result = worker_function(worker_context, *task_data[1:])

            result_queue.put((task_id, result))
            logger.info(f"{worker_name} task {task_id} completed")

        except Exception as e:
            logger.error(f"Error in {worker_name} worker: {e}")
            if "task_id" in locals():
                error_queue.put((task_id, str(e)))

    logger.info(f"{worker_name} worker stopped")


class CpuWorkloadHandler:
    def __init__(
        self,
        name: str,
        worker_count: int,
        worker_function,
        worker_context_setup=None,
        warmup_task_data=None,
    ):
        self.name = name
        self.worker_count = worker_count
        self.worker_function = worker_function
        self.worker_context_setup = worker_context_setup
        self.logger = TTLogger()
        self._init_queues()

        self.result_futures = {}
        self.result_futures_lock = Lock()
        self.listener_running = True

        self._start_workers()
        self._warmup_workers(warmup_task_data)

        self.result_listener_task = asyncio.create_task(self._result_listener())
        self.error_listener_task = asyncio.create_task(self._error_listener())

    def _init_queues(self):
        self.task_queue = TTQueue(
            max_size=settings.max_queue_size, batch_enabled=settings.max_batch_size > 1
        )
        self.result_queue = Queue()
        self.error_queue = Queue()

    def _start_workers(self):
        """Start worker processes"""
        self.workers = []
        for i in range(self.worker_count):
            worker = Process(
                target=_process_worker_tasks,
                args=(
                    self.task_queue,
                    self.result_queue,
                    self.error_queue,
                    self.name,
                    i,
                    self.worker_function,
                    self.worker_context_setup,
                ),
                name=f"{self.name}Worker-{i}",
            )
            worker.start()
            self.workers.append(worker)
            self.logger.info(f"Started {self.name} worker {i} with PID {worker.pid}")
            time.sleep(settings.new_runner_delay_seconds)

    def _warmup_workers(self, warmup_task_data=None):
        if warmup_task_data is None:
            self.logger.info("No warmup task data provided, skipping warmup")
            return
        for i in range(self.worker_count):
            warmup_task_id = f"warmup-{i}"
            self.task_queue.put((warmup_task_id,) + tuple(warmup_task_data))
            self.logger.info(f"Submitted warmup task for {self.name} worker {i}")

        self.logger.info(
            f"Submitted {self.worker_count} warmup tasks to {self.name} workers"
        )

    async def _result_listener(self):
        """Listen for results from worker processes"""
        while self.listener_running:
            try:
                task_id, result = await asyncio.to_thread(self.result_queue.get)

                if task_id is None:
                    break

                with self.result_futures_lock:
                    future = self.result_futures.pop(task_id, None)

                if future and not future.cancelled():
                    future.set_result(result)

            except Exception as e:
                self.logger.error(f"Error in {self.name} result listener: {e}")

        self.logger.info(f"{self.name} result listener stopped")

    async def _error_listener(self):
        """Listen for errors from worker processes"""
        while self.listener_running:
            try:
                task_id, error = await asyncio.to_thread(self.error_queue.get)

                if task_id is None:  # Shutdown signal
                    break

                self.logger.error(f"{self.name} error for task {task_id}: {error}")

                with self.result_futures_lock:
                    future = self.result_futures.pop(task_id, None)

                if future and not future.cancelled():
                    future.set_exception(Exception(error))

            except Exception as e:
                self.logger.error(f"Error in {self.name} error listener: {e}")

        self.logger.info(f"{self.name} error listener stopped")

    def stop_workers(self):
        try:
            self.listener_running = False

            # Send shutdown signals to workers
            for _ in self.workers:
                try:
                    self.task_queue.put(None, timeout=2.0)
                except Exception:
                    self.logger.warning(
                        f"Timeout sending shutdown signal to {self.name} worker"
                    )

            # Send shutdown signals to listeners
            try:
                self.result_queue.put((None, None), timeout=1.0)
                self.error_queue.put((None, None), timeout=1.0)
            except Exception:
                self.logger.warning(
                    f"Timeout sending shutdown signals to {self.name} listeners"
                )

            # Wait for workers to finish
            for i, worker in enumerate(self.workers):
                if worker.is_alive():
                    worker.join(timeout=10.0)
                    if worker.is_alive():
                        self.logger.warning(
                            f"{self.name} worker {i} did not shutdown gracefully"
                        )
                        worker.terminate()
                        worker.join(timeout=2.0)
                        if worker.is_alive():
                            worker.kill()

            if hasattr(self, "result_listener_task"):
                self.result_listener_task.cancel()
            if hasattr(self, "error_listener_task"):
                self.error_listener_task.cancel()

            try:
                self.task_queue.close()
                self.result_queue.close()
                self.error_queue.close()
                self.task_queue.join_thread()
                self.result_queue.join_thread()
                self.error_queue.join_thread()
            except Exception as e:
                self.logger.error(f"Error closing {self.name} queues: {e}")

            with self.result_futures_lock:
                for task_id, future in self.result_futures.items():
                    if not future.done():
                        future.cancel()
                self.result_futures.clear()

            self.logger.info(f"{self.name} workers stopped")

        except Exception as e:
            self.logger.error(f"Error during {self.name} worker shutdown: {e}")

    async def execute_task(self, *task_args):
        """Submit a task to the worker queue and return the result asynchronously"""
        task_id = str(uuid.uuid4())

        loop = asyncio.get_event_loop()
        result_future = loop.create_future()

        with self.result_futures_lock:
            self.result_futures[task_id] = result_future

        task_data = (task_id,) + task_args
        self.task_queue.put(task_data)

        try:
            return await asyncio.wait_for(result_future, timeout=None)
        except Exception:
            self._pop_and_cancel_future(task_id)
            raise

    def _pop_and_cancel_future(self, key):
        with self.result_futures_lock:
            future = self.result_futures.pop(key, None)
            if future and not future.done():
                future.cancel()
