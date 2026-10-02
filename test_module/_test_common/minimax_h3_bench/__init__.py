# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The MiniMax-H3 generation benchmark, vendored from quad-agent/h3-benchmark.

Drives a tt-media-server MiniMax-H3 deployment through its V1 job API with the
21-case matrix in ``cases.json`` (text-to-video, first/last-frame, omni-reference
at 5/10/15 s), records one row per generation, and judges every clip against the
output contract (24 fps, 17n+5 frames, the 16:9 canvas, an audible soundtrack that
is not at the rails). Standard library plus ffmpeg/ffprobe.

Modules: ``models`` (paths, tunables, timeout table, classification), ``adapters``
(HTTP + the tt-h3 request shapes), ``runner`` (submit -> poll -> fetch, retries,
resume, verdicts), ``judge`` (the clip contract), ``host`` (endpoint discovery,
health pre-gate, strikes/wedge state, cancel cleanup, one case end to end).
"""

from . import adapters, host, judge, models, runner

__all__ = ["adapters", "host", "judge", "models", "runner"]
