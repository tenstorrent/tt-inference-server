# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import io
import requests
from PIL import Image
from locust import HttpUser, task

# Save image as JPEG in-memory for load testing
# Load sample image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
pil_image = Image.open(requests.get(url, stream=True).raw)
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
