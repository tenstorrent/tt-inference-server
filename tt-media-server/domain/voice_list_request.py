# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from domain.base_request import BaseRequest


class VoiceListRequest(BaseRequest):
    """Marker request for listing registered voice-clone voices.

    Carries no fields -- it exists so a ``GET /v1/audio/voices`` call flows
    through the exact same Scheduler/device_worker IPC path as every other
    request. The voice-clone cache lives inside the runner's own process (not
    reachable directly from the FastAPI process), so the list must be produced
    on the worker side and returned across the IPC boundary.
    """

    pass
