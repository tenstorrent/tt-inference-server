# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import torch


def set_torch_thread_limits(num_threads: int = 1):
    if torch.get_num_threads() != num_threads:
        torch.set_num_threads(num_threads)
    if torch.get_num_interop_threads() != num_threads:
        torch.set_num_interop_threads(num_threads)
