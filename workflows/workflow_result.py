# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowResult:
    workflow_name: str
    return_code: int
