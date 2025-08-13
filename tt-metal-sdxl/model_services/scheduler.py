# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import asyncio
from multiprocessing import Process, Queue as Queue  # Need multiprocessing queues
from threading import Lock

from fastapi import HTTPException
from config.settings import Settings, get_settings
from model_services.device_worker import device_worker
from tt_model_runners.runner_fabric import get_device_runner
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
        # For multiprocessing, need multiprocessing queues
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
        # locks
        self.ready_devices_lock = Lock()
        self.result_futures_lock = Lock()
        # main device holder
        self.main_device = None

    def is_queue_full(self):
        return self.task_queue.full()

    @log_execution_time("Scheduler image processing")
    def process_request(self, request):
        try:
            self.checkIsModelReady()
            
            if self.task_queue.full():
                raise HTTPException(
                    status_code=429, 
                    detail="Task queue is full. Please try again later."
                )
            
            # Non-blocking put with timeout
            try:
                self.task_queue.put(request, timeout=1.0)
            except:
                raise HTTPException(
                    status_code=429,
                    detail="Unable to queue request - system busy"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail="Internal error processing request"
            )

    def checkIsModelReady(self) -> bool:
        if (self.isReady is not True):
            raise HTTPException(405, "Model is not ready")
        return True

    @log_execution_time("Scheduler image processing")
    def startWorkers(self):
        # keep result listener in the main event loop
        self.listener_task_ref = asyncio.create_task(self.result_listener())

        # keep device warmup listener in the main event loop, it'll close soon
        self.device_warmup_listener_ref = asyncio.create_task(self.device_warmup_listener())
        # keep error listener in the main event loop
        self.error_queue_listener_ref = asyncio.create_task(self.error_listener())

        for i in range(self.worker_count):
            p = Process(
                target=device_worker, 
                args=(i, self.task_queue, self.result_queue, self.warmup_signals_queue, self.error_queue),
                name=f"DeviceWorker-{i}"
            )
            p.start()
            self.workers.append(p)
        self.logger.info(f"Workers started: {1}")

    async def result_listener(self):
        while self.listener_running:
            try:
                task_id, image = await asyncio.to_thread(self.result_queue.get)
                if task_id is None:
                    self.listener_running = False
                    break
                
                # Thread-safe access to futures
                with self.result_futures_lock:
                    future = self.result_futures.pop(task_id, None)
                
                if future and not future.cancelled():
                    future.set_result(image)
                elif not future:
                    self.logger.warning(f"No future found for task {task_id}")
                    
            except Exception as e:
                self.logger.error(f"Error in result_listener: {e}", exc_info=True)
        
        self.warmup_signals_queue.put(None, timeout=1.0)
        self._close_queues([self.warmup_signals_queue])
        self.logger.info("Result listener stopped")

    async def error_listener(self):
        while self.listener_running:
            try:
                task_id, error = await asyncio.to_thread(self.error_queue.get)
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
                with self.ready_devices_lock:
                    if device_id not in self.ready_devices:
                        self.ready_devices.append(device_id)
                        # Set ready as soon as first device is available
                        if not self.isReady:
                            self.isReady = True
                        
                        if len(self.ready_devices) == len(self.workers):
                            self.logger.info("All devices are warmed up and ready")
                            self.device_warmup_listener_running = False
            
            except Exception as e:
                self.logger.error(f"Error in device_warmup_listener: {e}", exc_info=True)
        
        self.logger.info("Device warmup listener is done")


    def stopWorkers(self):
        self.logger.info("Stopping workers")
        
        try:
            # Stop accepting new requests
            self.isReady = False
            
            # Send shutdown signals to all workers
            for _ in self.workers:
                try:
                    self.task_queue.put(None, timeout=2.0)
                except:
                    self.logger.warning("Timeout sending shutdown signal to worker")
            
            # Wait for processes to finish gracefully
            for i, worker in enumerate(self.workers):
                if worker.is_alive():
                    worker.join(timeout=10.0)  # Increased timeout
                    if worker.is_alive():
                        self.logger.warning(f"Worker {i} did not shutdown gracefully")
                        worker.terminate()  # Terminate process (not kill)
                        worker.join(timeout=2.0)  # Wait for termination
                        if worker.is_alive():
                            worker.kill()  # Force kill as last resort
            
            self.workers.clear()
            
            # Stop listeners
            self.listener_running = False
            self.device_warmup_listener_running = False
            
            # Send shutdown signals to listeners
            try:
                self.result_queue.put((None, None), timeout=1.0)
                self.error_queue.put((None, None), timeout=1.0)
                self.warmup_signals_queue.put(None, timeout=1.0)
            except:
                self.logger.warning("Timeout sending shutdown signals to listeners")

            # close queues
            self._close_queues(
                [self.task_queue, 
                 self.result_queue, 
                 self.warmup_signals_queue, 
                 self.error_queue])

            # Cancel any remaining futures
            with self.result_futures_lock:
                for task_id, future in self.result_futures.items():
                    if not future.done():
                        future.cancel()
                        self.logger.info(f"Cancelled pending task {task_id}")
                self.result_futures.clear()
            
            # Clear device state
            with self.ready_devices_lock:
                self.ready_devices.clear()
            
            # close device
            if self.main_device:
                get_device_runner(0).close_device(self.main_device)
                self.logger.info("Main device closed")
            
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