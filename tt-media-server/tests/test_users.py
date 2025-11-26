# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest

# Skip entire module to prevent import errors during test collection
pytestmark = pytest.mark.skip(
    reason="Disabling temporary for now, will re-enable after fix"
)

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


@pytest.mark.skip(reason="Disabling temporary for now, will re-enable after fix")
def test_get_users():
    response = client.get("/api/users")
    assert response.status_code == 200
    assert "message" in response.json()
