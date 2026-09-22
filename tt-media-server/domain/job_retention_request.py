# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from pydantic import BaseModel


class JobRetentionRequest(BaseModel):
    retained: bool
