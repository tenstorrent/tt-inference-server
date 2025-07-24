# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import abstractmethod

class DeviceRunner:

    @abstractmethod
    def load_model(self):
        raise NotImplementedError()

    @abstractmethod
    def runInference(self, prompt: str, num_inference_steps: int = 50):
        raise NotImplementedError()

    @abstractmethod
    def close_device(self):
        raise NotImplementedError()