# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from http import HTTPStatus
import os
import requests
import time
from utils import get_auth_header, get_sample_prompt


DEPLOY_URL = "http://127.0.0.1"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 7000))
API_BASE_URL = f"{DEPLOY_URL}:{SERVICE_PORT}"
API_URL = f"{API_BASE_URL}/submit"
API_GET_IMAGE_URL = f"{API_BASE_URL}/get_image"
API_GET_LATEST_PROMPT_URL = f"{API_BASE_URL}/get_latest_time"
HEALTH_URL = f"{API_BASE_URL}/health"


def test_valid_api_call():
    # get sample prompt
    sample_prompt = get_sample_prompt()
    body = {"prompt": sample_prompt}
    # make request with auth headers
    headers = get_auth_header()
    response = requests.post(API_URL, json=body, headers=headers)
    # perform status and value checking
    assert response.status_code == HTTPStatus.OK
    assert isinstance(response.json(), dict)

    # wait generous amount of time for image to be generated
    time.sleep(15)
    # request image
    response = requests.get(API_GET_IMAGE_URL)
    assert response.status_code == HTTPStatus.OK

    # check that prompt was correctly generated
    response = requests.get(API_GET_LATEST_PROMPT_URL)
    assert response.status_code == HTTPStatus.OK
    assert response.json()["prompt"] == sample_prompt


def test_invalid_api_call():
    # get sample prompt
    sample_prompt = get_sample_prompt()
    body = {"prompt": sample_prompt}
    # make request with INVALID auth header
    headers = get_auth_header()
    headers.update(Authorization="INVALID API KEY")
    response = requests.post(API_URL, json=body, headers=headers)
    # assert request was unauthorized
    assert response.status_code == HTTPStatus.UNAUTHORIZED


def test_get_health():
    headers = {}
    response = requests.get(HEALTH_URL, headers=headers, timeout=5)
    assert response.status_code == 200
