# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Shared authentication helpers for test/workflow runners.

Centralizes how ``OPENAI_API_KEY`` is derived for pytest subprocesses so the
same remote-vs-local auth rule is applied everywhere instead of being
re-implemented inline in each runner.
"""

from __future__ import annotations

import json
import os

import jwt


def setup_tests_auth(jwt_secret: str, remote_server: bool, logger) -> None:
    """Configure OPENAI_API_KEY for pytest subprocesses.

    Remote (--server-url): literal API_KEY / OPENAI_API_KEY only.
    Local: JWT_SECRET (standard workflow auth) or literal key fallback.
    """
    if remote_server:
        literal_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        if not literal_key:
            logger.warning(
                "No API_KEY or OPENAI_API_KEY set; remote endpoint requests "
                "will likely fail with 401."
            )
            return
        os.environ["OPENAI_API_KEY"] = literal_key
        logger.info(
            "OPENAI_API_KEY set from API_KEY / OPENAI_API_KEY for remote tests."
        )
        return

    if jwt_secret:
        json_payload = json.loads(
            '{"team_id": "tenstorrent", "token_id": "debug-test"}'
        )
        encoded_jwt = jwt.encode(json_payload, jwt_secret, algorithm="HS256")
        os.environ["OPENAI_API_KEY"] = encoded_jwt
        logger.info(
            "OPENAI_API_KEY environment variable set using provided JWT secret."
        )
    elif os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"):
        literal_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = literal_key
        logger.info(
            "OPENAI_API_KEY environment variable set from literal "
            "API_KEY / OPENAI_API_KEY."
        )
