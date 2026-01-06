# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.scheduler import Scheduler
import threading

# scheduler is singleton
_current_scheduler_holder = None
_scheduler_lock = threading.Lock()


def get_scheduler() -> Scheduler:
    global _current_scheduler_holder
    with _scheduler_lock:
        if _current_scheduler_holder is None:
            _current_scheduler_holder = Scheduler()
    return _current_scheduler_holder
