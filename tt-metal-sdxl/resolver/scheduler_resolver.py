# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from model_services.scheduler import Scheduler

# scheduler is singleton
_current_scheduler_holder = None

def get_scheduler() -> Scheduler:
    global _current_scheduler_holder
    if _current_scheduler_holder is None:
        _current_scheduler_holder = Scheduler()
    return _current_scheduler_holder