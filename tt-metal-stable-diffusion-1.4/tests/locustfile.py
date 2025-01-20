# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

from locust import HttpUser, task
from utils import get_auth_header, sample_file


# load sample file in memory
file = sample_file()


class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        headers = get_auth_header()
        self.client.post("/objdetection_v2", files=file, headers=headers)
