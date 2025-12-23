# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
import os
import time
from multiprocessing import Process  # Need multiprocessing queues
from multiprocessing import Queue as Queue
from threading import Lock

from config.settings import get_settings
from fastapi import HTTPException
from model_services.device_worker import device_worker
from model_services.tt_queue import TTQueue
from utils.helpers import log_execution_time
from utils.logger import TTLogger


class Scheduler:
    @log_execution_time("Scheduler init")
    def __init__(self):
        self.settings = get_settings()
        self.logger = TTLogger()
        self._setup_initial_variables()
        self._start_queues()

    def _start_queues(self):
        worker_count = self.get_worker_count()
        self.task_queue = TTQueue(
            self.settings.max_queue_size, batch_enabled=self.settings.max_batch_size > 1
        )
        self.warmup_signals_queue = Queue(worker_count)
        self.result_queue = Queue()
        self.error_queue = Queue()

    def get_worker_count(self):
        if not hasattr(self, "worker_count"):
            self.worker_count = self._calculate_worker_count()
        return self.worker_count

    def _setup_initial_variables(self):
        self.isReady = False
        self.listener_running = True
        self.device_warmup_listener_running = True
        self.workers_to_open = []
        self.worker_info = {}
        self.monitor_running = True
        self.result_futures = {}
        # locks
        self.result_futures_lock = Lock()
        # Task references for asyncio tasks
        self.monitor_task_ref = None
        self.listener_task_ref = None
        self.device_warmup_listener_ref = None
        self.error_queue_listener_ref = None

    @log_execution_time("Scheduler request processing")
    def process_request(self, request):
        try:
            self.check_is_model_ready()

            if self.task_queue.full():
                raise HTTPException(
                    status_code=429,
                    detail="Task queue is full. Please try again later.",
                )

            # Non-blocking put with timeout
            try:
                self.task_queue.put(request, timeout=1.0)
            except Exception:
                raise HTTPException(
                    status_code=429, detail="Unable to queue request - system busy"
                )

        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, detail="Internal error processing request"
            )

    def check_is_model_ready(self) -> bool:
        if self.isReady is not True:
            raise HTTPException(405, "Model is not ready")
        return True

    @log_execution_time("Scheduler - starting workers")
    def start_workers(self):
        # keep result listener in the main event loop
        self.listener_task_ref = asyncio.create_task(self.result_listener())

        # keep device warmup listener in the main event loop, it'll close soon
        self.device_warmup_listener_ref = asyncio.create_task(
            self.device_warmup_listener()
        )
        # keep error listener in the main event loop
        self.error_queue_listener_ref = asyncio.create_task(self.error_listener())

        self.logger.info(f"Workers to start: {self.worker_count}")
        asyncio.create_task(self._start_workers_in_sequence())

    async def _start_workers_in_sequence(self):
        """Start workers one by one with a delay to avoid overload"""
        while self.workers_to_open:
            self._start_worker()
            self.logger.info(
                f"Worker started, remaining workers to open: {len(self.workers_to_open)}"
            )

            # Add delay between worker starts to avoid resource contention
            if self.workers_to_open:  # Only sleep if there are more workers to start
                await asyncio.sleep(self.settings.new_device_delay_seconds)

        self.logger.info("All workers started in sequence")

    def _start_worker(self, worker_id=None):
        """Start a single worker process"""
        if worker_id is None:
            worker_id = (
                self.workers_to_open.pop(0)
                if self.workers_to_open
                else Exception("No more workers to start")
            )
            # in case it's a device pair remove starting bracket open
            worker_id = worker_id.lstrip("(").rstrip(")")
        self.logger.info(f"Starting worker {worker_id}")
        p = Process(
            target=device_worker,
            args=(
                worker_id,
                self.task_queue,
                self.result_queue,
                self.warmup_signals_queue,
                self.error_queue,
            ),
            name=f"DeviceWorker-{worker_id}",
        )
        p.start()

        self.worker_info[worker_id] = {
            "process": p,
            "start_time": time.time(),
            "restart_count": 0,
            "is_ready": False,
            "error_count": 0,
        }

        self.logger.info(f"Started worker {worker_id} with PID {p.pid}")

    def restart_worker(self, worker_id: str):
        """Restart a dead worker"""
        old_info = self.worker_info.get(worker_id, {})

        if old_info == {}:
            raise ValueError(f"Worker ID {worker_id} not found in worker info")

        restart_count = old_info.get("restart_count", 0) + 1

        self.logger.warning(
            f"Restarting dead worker {worker_id} (restart #{restart_count})"
        )

        # Clean up old process if it exists
        if worker_id in self.worker_info:
            try:
                old_process = self.worker_info[worker_id]["process"]
                if old_process.is_alive():
                    old_process.terminate()
                    old_process.join(timeout=5.0)
            except Exception as e:
                self.logger.error(f"Error cleaning up old worker {worker_id}: {e}")
                self.logger.info(f"Old worker {worker_id} process does not exist")

        # Start new worker
        self._start_worker(worker_id)
        self.worker_info[worker_id]["restart_count"] = restart_count
        # pass the error count from old worker -1 to give it a chance to recover
        self.worker_info[worker_id]["error_count"] = old_info.get("error_count", 1) - 1

    async def result_listener(self):
        while self.listener_running:
            try:
                worker_id, task_id, input = await asyncio.to_thread(
                    self.result_queue.get
                )

                if task_id is None:
                    self.listener_running = False
                    break

                with self.result_futures_lock:
                    future = self.result_futures.pop(task_id, None)

                if future and not future.cancelled():
                    future.set_result(input)
                elif not future:
                    current_futures = list(self.result_futures.keys())
                    self.logger.warning(
                        f"No future found for task {task_id}. Current futures: {current_futures}"
                    )

                # Reset worker restart count on successful job
                self.worker_info[worker_id]["restart_count"] = 0

            except Exception as e:
                self.logger.error(f"Error in result_listener: {e}", exc_info=True)

        self.logger.info("Result listener stopped")

    async def error_listener(self):
        while self.listener_running:
            try:
                worker_id, task_id, error = await asyncio.to_thread(
                    self.error_queue.get
                )

                self.worker_info[worker_id]["error_count"] += 1

                self.logger.warning(
                    f"Worker {worker_id} error count is : {self.worker_info[worker_id]['error_count']}"
                )

                if task_id is None:
                    self.listener_running = False
                    break

                self.logger.error(f"Error in worker {task_id}: {error}")

                # Thread-safe future handling
                with self.result_futures_lock:
                    future = self.result_futures.pop(task_id, None)

                if future and not future.cancelled():
                    future.set_exception(Exception(error))

            except Exception as e:
                self.logger.error(f"Error in error_listener: {e}", exc_info=True)

        self.logger.info("Error listener stopped")

    async def device_warmup_listener(self):
        while self.device_warmup_listener_running:
            try:
                device_id = await asyncio.to_thread(self.warmup_signals_queue.get)
                if device_id is None:  # Shutdown signal
                    break

                self.logger.info(f"Device {device_id} is warmed up")

                # Thread-safe device tracking
                self.worker_info[device_id]["is_ready"] = True
                self.worker_info[device_id]["ready_time"] = time.time()
                # Set ready as soon as first device is available
                if not self.isReady:
                    self.isReady = True

                    self.logger.info(
                        "First device warmed up, starting worker health monitor"
                    )
                    self.monitor_task_ref = asyncio.create_task(
                        self.worker_health_monitor()
                    )

                all_devices_ready = all(
                    info["is_ready"] for info in self.worker_info.values()
                )
                if all_devices_ready:
                    self.logger.info("All devices are warmed up and ready")

            except Exception as e:
                self.logger.error(
                    f"Error in device_warmup_listener: {e}", exc_info=True
                )

        self.logger.info("Device warmup listener is done")

    @log_execution_time("Scheduler - stopping workers")
    def stop_workers(self):
        self.logger.info("Stopping workers")

        try:
            # Stop monitoring
            self.monitor_running = False
            if self.monitor_task_ref:
                self.monitor_task_ref.cancel()

            # Stop accepting new requests
            self.isReady = False

            # Send shutdown signals to all workers
            for _ in self.worker_info:
                try:
                    self.task_queue.put(None, timeout=2.0)
                except:
                    self.logger.warning("Timeout sending shutdown signal to worker")

            # Wait for processes to finish gracefully
            for i, worker_element in self.worker_info.items():
                worker = worker_element["process"]
                if worker.is_alive():
                    worker.join(timeout=10.0)  # Increased timeout
                    if worker.is_alive():
                        self.logger.warning(f"Worker {i} did not shutdown gracefully")
                        worker.terminate()  # Terminate process (not kill)
                        worker.join(timeout=2.0)  # Wait for termination
                        if worker.is_alive():
                            worker.kill()  # Force kill as last resort

            self.worker_info = {}  # Clear worker info

            self.logger.info("All workers stopped successfully")

            # Stop listeners
            self.listener_running = False
            self.device_warmup_listener_running = False

            # Send shutdown signals to listeners
            try:
                self.result_queue.put((None, None, None), timeout=1.0)
                self.error_queue.put((None, None, None), timeout=1.0)
                self.warmup_signals_queue.put(None, timeout=1.0)
            except:
                self.logger.warning("Timeout sending shutdown signals to listeners")

            # close queues
            self._close_queues(
                [
                    self.task_queue,
                    self.result_queue,
                    self.warmup_signals_queue,
                    self.error_queue,
                ]
            )

            self.logger.info("Queues closed successfully")

            # Cancel any remaining futures
            with self.result_futures_lock:
                for task_id, future in self.result_futures.items():
                    if not future.done():
                        future.cancel()
                        self.logger.info(f"Cancelled pending task {task_id}")
                self.result_futures.clear()

            self.logger.info("Workers stopped")

        except Exception as e:
            self.logger.error(f"Error during worker shutdown: {e}", exc_info=True)

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

    def _calculate_worker_count(self) -> int:
        try:
            device_ids_cleaned = self.settings.device_ids.replace(" ", "").split("),(")
            worker_count = len(device_ids_cleaned)
            self.workers_to_open = device_ids_cleaned
            from loguru import logger
            logger.warning(f"{self.workers_to_open=}")
            if worker_count < 1:
                self.logger.error("Worker count is 0")
                raise ValueError("Worker count must be at least 1")
            return worker_count
        except Exception as e:
            self.logger.error(f"Error getting workers count: {e}")
            raise HTTPException(status_code=500, detail="Workers cannot be initialized")

    def _get_max_queue_size(self) -> int:
        try:
            max_queue_size = self.settings.max_queue_size
            if max_queue_size < 1:
                self.logger.error("Max queue size is 0")
                raise ValueError("Max queue size must be at least 1")
            return max_queue_size
        except Exception as e:
            self.logger.error(f"Error getting max queue size: {e}")
            raise HTTPException(
                status_code=500, detail="Max queue size not provided in settings"
            )

    async def worker_health_monitor(self):
        """Monitor worker health and restart dead workers"""
        while self.monitor_running and self.isReady:
            try:
                dead_workers = []

                for worker_id, info in self.worker_info.items():
                    try:
                        process = info["process"]
                        if not process.is_alive():
                            dead_workers.append(worker_id)
                    except Exception as e:
                        self.logger.error(
                            f"Error checking worker {worker_id} health: {e}"
                        )
                        dead_workers.append(worker_id)

                # check for any workers that have too many errors
                for worker_id, info in self.worker_info.items():
                    if info["error_count"] > self.settings.max_worker_restart_count:
                        dead_workers.append(worker_id)
                        self.logger.error(
                            f"Worker {worker_id} has too many errors ({info['error_count']}), restarting"
                        )

                self.logger.info(
                    f"Worker health check: {len(dead_workers)} dead workers found"
                )
                # Restart dead workers
                for worker_id in dead_workers:
                    restart_count = self.worker_info[worker_id].get("restart_count", 0)

                    # Optional: Limit restart attempts
                    if (
                        restart_count < self.settings.max_worker_restart_count
                    ):  # Max 5 restarts per worker
                        self.restart_worker(worker_id)
                    else:
                        self.logger.error(
                            f"Worker {worker_id} has died too many times ({restart_count}), restart did not help"
                        )
                        if self.settings.allow_deep_reset:
                            self.logger.info("Trying deep restart of all workers")
                            self.deep_restart_workers()

                await asyncio.sleep(self.settings.worker_check_sleep_timeout)

            except Exception as e:
                self.logger.error(f"Error in worker_health_monitor: {e}", exc_info=True)
                await asyncio.sleep(1.0)

        self.logger.info("Worker health monitor stopped")

    async def deep_restart_workers(self):
        """Restart all workers"""
        self.logger.info("Deep restarting all workers")

        # Stop current workers
        self.stop_workers()

        # try to reset the device
        exit_code = os.system(self.settings.reset_device_command)

        # Clear worker info
        self.worker_info.clear()

        self.logger.info(f"Reset command executed with exit code: {exit_code}")

        # Wait for a short period to ensure all processes are cleaned up
        await asyncio.sleep(self.settings.reset_device_sleep_time)

        self.logger.info("Restarting queues after reset")

        self._setup_initial_variables()
        self._start_queues()

        self.logger.info("Starting new workers after reset")

        # Start new workers
        self.start_workers()

        self.logger.info("All workers restarted successfully")

    def get_worker_info(self) -> dict:
        """Get serializable worker information for monitoring"""
        serializable_worker_info = {}
        for worker_id, info in self.worker_info.items():
            serializable_worker_info[worker_id] = {
                "pid": info["process"].pid if info["process"].is_alive() else None,
                "is_alive": info["process"].is_alive(),
                "start_time": info["start_time"],
                "is_ready": info["is_ready"],
                "restart_count": info["restart_count"],
                "error_count": info["error_count"],
                "ready_time": info["ready_time"] if "ready_time" in info else None,
            }
        return serializable_worker_info

    def pop_and_cancel_future(self, key):
        """Thread-safe removal and cancellation of a future from result_futures."""
        with self.result_futures_lock:
            future = self.result_futures.pop(key, None)
            if future and not future.done():
                future.cancel()
