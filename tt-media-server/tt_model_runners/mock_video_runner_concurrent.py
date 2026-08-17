# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Concurrent T2V mock runner — drop-in replacement for the SHM bridge.

Processes multiple in-flight T2V requests in parallel, with no per-request
validation. The ``image_path`` field on the SHM ``VideoRequest`` is
deliberately ignored: this mock answers every request as if it were T2V,
which is useful for perf testing the SHM ring + worker pool + encoder
hand-off split independently of the I2V side-file machinery.

For an I2V-strict counterpart (validates the side-file end-to-end and
returns an ERROR response on contract violations), use
``mock_video_runner_concurrent_i2v.py`` instead.

Shared scaffolding lives in :mod:`mock_video_runner_base`; this file
only owns the entrypoint.

Expected steady-state timing for 8 requests at L=2, E=1, C=1:
    wall-clock ≈ 8·L + 1·E = 17 s   (NOT 8·(L+E) = 24 s)
If you observe 24 s, the encoder split has regressed.

Usage::

    TT_VIDEO_SHM_INPUT=tt_video_in TT_VIDEO_SHM_OUTPUT=tt_video_out \\
    MOCK_CONCURRENCY=8 MOCK_LATENCY_S=2.0 MOCK_ENCODE_S=1.0 \\
        python -m tt_model_runners.mock_video_runner_concurrent
"""

from __future__ import annotations

import signal

from tt_model_runners.mock_video_runner_base import handleSignal, runMockBridge

LABEL = "[T2V-MOCK]"


def main() -> None:
    signal.signal(signal.SIGTERM, handleSignal)
    signal.signal(signal.SIGINT, handleSignal)
    runMockBridge(label=LABEL)


if __name__ == "__main__":
    main()
