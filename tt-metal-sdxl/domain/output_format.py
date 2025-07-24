# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from enum import Enum

class OutputFormat(str, Enum):
    FILE = "FILE"
    BASE_64 = "BASE_64"