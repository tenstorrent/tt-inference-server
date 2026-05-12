# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Docker-image interface eras.

The inference-server image's docker-run contract changed in v0.11.0 (commit
50db8ac7, "Simplify and improve vLLM Docker image interface"). Before that
release, the image used a docker-entrypoint.sh wrapper that exec'd via gosu
(a small Docker-aware su tool) and took no CLI args; after that release, the
image uses a self-contained bash-c ENTRYPOINT that accepts `--model`,
`--tt-device`, and other flags after the image name.

This module is the single source of truth for which `model_spec.version`
maps to which era. It's intentionally lightweight (no heavy imports) so any
caller can ask "what era is this version?" without pulling in the rest of
the workflow stack.

Per-era differences in `docker run` (see workflows/run_docker_server.py
for the branching that implements them):

    field               V1_LEGACY (< 0.11.0)           V2_MODERN (>= 0.11.0)
    ------------------  -----------------------------  -----------------------------
    ENTRYPOINT          docker-entrypoint.sh (gosu)    bash -c source venv && python
                                                       run_vllm_api_server.py "$@"
    args/CMD shape      bash -c "...python ...args"    --model X --tt-device Y ...
                        as a CMD override              as raw CMD ($@)
    shared memory       --shm-size 32G                 --ipc host
    spec JSON mount     required (image has no         --dev-mode only (image
                        baked-in catalog)              has baked-in catalog)

The in-image script (run_vllm_api_server.py) is the same across both eras —
v0.10.0 images were built from main after commit 50db8ac7 (which rewrote the
script) but before the Dockerfile ENTRYPOINT refactor. Env var names
(RUNTIME_MODEL_SPEC_JSON_PATH, MODEL_WEIGHTS_DIR, CACHE_ROOT, TT_CACHE_PATH)
are therefore identical across eras.
"""

from enum import Enum
from typing import List, Tuple

from workflows.utils import parse_version_tuple


class DockerInterface(Enum):
    """Era of the inference-server image's docker-run contract.

    To add a new era when the interface breaks again (e.g. at v0.20.0):
      1. Add a new enum value below.
      2. Prepend the cut tuple to _DOCKER_INTERFACE_ERAS.
      3. Add the corresponding branch in `generate_docker_run_command` for
         every field that differs (see the table in this module's docstring).

    The "data-only change" is just step 2; new behavior still needs code at
    each divergence point.
    """

    V1_LEGACY = "v1-legacy"  # pre-0.11.0
    V2_MODERN = "v2-modern"  # >= 0.11.0


# Minimum version that selected v0.11.0 — the first release shipping the new
# bash-c ENTRYPOINT (commit 50db8ac7 landed in v0.11.0).
_V2_MIN_VERSION: Tuple[int, int, int] = (0, 11, 0)


# Source of truth: which image version maps to which era. Each entry is
# (min_version_inclusive, era). Newest era FIRST; first row whose min_version
# <= image_version wins. Unparseable image tags (`:dev`, `:latest`, missing)
# default to the newest era — see get_docker_interface().
_DOCKER_INTERFACE_ERAS: List[Tuple[Tuple[int, int, int], DockerInterface]] = [
    (_V2_MIN_VERSION, DockerInterface.V2_MODERN),
    ((0, 0, 0), DockerInterface.V1_LEGACY),
]


def get_docker_interface(version: str) -> DockerInterface:
    """Return the docker-interface era for a semver-ish version string.

    Designed for ``model_spec.version`` — the authoritative per-template
    release version. Unparseable versions fall back to the newest era so
    today's behaviour on main (which assumes the V2_MODERN contract for
    every image) is preserved.

    Examples:
        >>> get_docker_interface("0.10.4")
        <DockerInterface.V1_LEGACY: 'v1-legacy'>
        >>> get_docker_interface("0.13.0")
        <DockerInterface.V2_MODERN: 'v2-modern'>
        >>> get_docker_interface("")
        <DockerInterface.V2_MODERN: 'v2-modern'>
    """
    assert _DOCKER_INTERFACE_ERAS, "era table must not be empty"
    parsed = parse_version_tuple(version)
    if parsed is None:
        return _DOCKER_INTERFACE_ERAS[0][1]
    for min_v, interface in _DOCKER_INTERFACE_ERAS:
        if parsed >= min_v:
            return interface
    # Unreachable while the table includes (0, 0, 0); the assert above and
    # the test_eras_table_covers_zero invariant guarantee it.
    raise RuntimeError(f"version {version!r} parsed to {parsed} but no era matched")
