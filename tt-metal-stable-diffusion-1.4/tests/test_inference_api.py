# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

from http import HTTPStatus
import os
import pytest
import requests
from utils import get_auth_header, get_sample_prompt


DEPLOY_URL = "http://127.0.0.1"
SERVICE_PORT = int(os.getenv("SERVICE_PORT", 7000))
API_BASE_URL = f"{DEPLOY_URL}:{SERVICE_PORT}"
API_URL = f"{API_BASE_URL}/submit"
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


@pytest.mark.skip(
    reason="Not implemented, see https://github.com/tenstorrent/tt-inference-server/issues/63"
)
def test_get_health():
    headers = {}
    response = requests.get(HEALTH_URL, headers=headers, timeout=35)
    assert response.status_code == 200
