# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os


def get_auth_header():
    if authorization_header := os.getenv("AUTHORIZATION", None):
        headers = {"Authorization": authorization_header}
        return headers
    else:
        raise RuntimeError("AUTHORIZATION environment variable is undefined.")


def get_sample_prompt():
    return "Red dog"
