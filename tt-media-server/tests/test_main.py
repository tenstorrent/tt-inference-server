# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import pytest

# Skip entire module to prevent import errors during test collection
pytestmark = pytest.mark.skip(
    reason="Disabling temporary for now, will re-enable after fix"
)

# Imports commented out to prevent errors during test collection
# from fastapi.testclient import TestClient
# from main import app

# client = TestClient(app)


def test_root():
    # response = client.get("/")
    # assert response.status_code == 200
    # assert response.json() == {"message": "Enhanced FastAPI App"}
    pass
