# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from abc import ABC
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, PrivateAttr


class BaseRequest(BaseModel, ABC):
    _task_id: str = PrivateAttr(default_factory=lambda: str(uuid4()))
    # Wall-clock stamps for cross-process perf (pickled through multiprocessing.Queue).
    _queue_wall_time: Optional[float] = PrivateAttr(default=None)
    _job_process_start_wall_time: Optional[float] = PrivateAttr(default=None)
