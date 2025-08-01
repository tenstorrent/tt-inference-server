# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import abstractmethod
from asyncio import Queue
from domain.image_generate_request import ImageGenerateRequest


class BaseModel:
    task_queue = Queue()
    result_futures = {}

    @abstractmethod
    def processImage(self, imageGenerateRequest: ImageGenerateRequest):
        pass

    @abstractmethod
    def checkIsModelReady(self) -> bool:
        pass

    @abstractmethod
    async def warmupModel(self):
        pass

    @abstractmethod
    def completions(self):
        pass

    @abstractmethod
    def startWorkers(self):
        pass

    @abstractmethod
    def stopWorkers(self):
        pass