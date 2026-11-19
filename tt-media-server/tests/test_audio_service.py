# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from model_services.audio_service import audio_worker_function


@pytest.mark.parametrize(
    "server_allows, request_enables, reason",
    [
        (False, True, "disabled on this server"),
        (True, False, "disabled for this request"),
    ],
)
def test_diarization_without_preprocessing_raises(
    server_allows, request_enables, reason
):
    manager = Mock()
    with patch(
        "config.settings.settings",
        SimpleNamespace(allow_audio_preprocessing=server_allows),
    ):
        with pytest.raises(ValueError, match=reason):
            audio_worker_function(
                manager, b"", request_enables, perform_diarization=True
            )
    manager.to_audio_array.assert_not_called()
