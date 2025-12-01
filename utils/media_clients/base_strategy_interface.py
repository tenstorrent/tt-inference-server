# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import ABC, abstractmethod
from .test_status import BaseTestStatus


class BaseMediaStrategy(ABC):
    """Interface for media strategies."""
    def __init__(self, all_params, model_spec, device, output_path, service_port, model_source="huggingface"):
        self.all_params = all_params
        self.model_spec = model_spec
        self.device = device
        self.output_path = output_path
        self.service_port = service_port
        self.model_source = model_source
        self.base_url = f"http://localhost:{service_port}"
        self.test_payloads_path = "utils/test_payloads"

    @abstractmethod
    def run_eval(self) -> None:
        """Run evaluation workflow for this media type."""
        pass

    @abstractmethod
    def run_benchmark(self, num_calls: int) -> list[BaseTestStatus]:
        """Run benchmark workflow for this media type."""
        pass

    @abstractmethod
    def get_health(self, attempt_number: int = 1) -> tuple[bool, str]:
        """Check health status for this media type with retry capability."""
        pass