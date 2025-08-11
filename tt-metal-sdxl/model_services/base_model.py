# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import ABC, abstractmethod
from asyncio import Queue
from domain.image_generate_request import ImageGenerateRequest

class BaseService(ABC):
    def __init__(self):
        self.task_queue = Queue()
        self.result_futures = {}

    @abstractmethod
    def process_image(self, image_generate_request: ImageGenerateRequest):
        pass

    @abstractmethod
    def check_is_model_ready(self) -> dict:
        pass

    @abstractmethod
    def start_workers(self):
        pass

    @abstractmethod
    def stop_workers(self):
        pass