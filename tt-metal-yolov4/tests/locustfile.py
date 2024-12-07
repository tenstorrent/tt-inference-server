# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import io
from PIL import Image
from locust import HttpUser, task

# Save image as JPEG in-memory for load testing
pil_image = Image.open("test_image.jpg")
pil_image = pil_image.resize((320, 320))  # Resize to target dimensions
buf = io.BytesIO()
pil_image.save(
    buf,
    format="JPEG",
)
byte_im = buf.getvalue()
file = {"file": byte_im}


class HelloWorldUser(HttpUser):
    @task
    def hello_world(self):
        self.client.post("/objdetection_v2", files=file)
