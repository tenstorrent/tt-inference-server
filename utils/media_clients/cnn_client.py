# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging

from .base_strategy_interface import BaseMediaStrategy

logger = logging.getLogger(__name__)


class CNNClient(BaseMediaStrategy):
    """Strategy for cnn models (RESNET, etc)."""

    def run_eval(self) -> None:
        pass

    def run_benchmark(self, num_calls):
        pass

    pass
