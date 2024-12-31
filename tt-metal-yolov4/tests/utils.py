# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import io
import os
from PIL import Image
import requests


def get_auth_header():
    if authorization_header := os.getenv("AUTHORIZATION", None):
        headers = {"Authorization": authorization_header}
        return headers
    else:
        raise RuntimeError("AUTHORIZATION environment variable is undefined.")


# save image as JPEG in-memory
def sample_file():
    # load sample image
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    pil_image = Image.open(requests.get(url, stream=True).raw)
    pil_image = pil_image.resize((320, 320))  # Resize to target dimensions
    # convert to bytes
    buf = io.BytesIO()
    # format as JPEG
    pil_image.save(
        buf,
        format="JPEG",
    )
    byte_im = buf.getvalue()
    file = {"file": byte_im}
    return file
