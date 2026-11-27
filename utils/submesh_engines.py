# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Submesh engines, shared by the vLLM launcher (vllm-tt-metal/src/run_vllm_api_server.py) and the host side.

A spec whose vllm_args set data_parallel_size N >= 2 and data_parallel_external_lb (vLLM's mode of one server per DP
rank behind an external load balancer) runs N independent single-rank vLLM engines, one per device group, behind
vllm-router on the service port. Engine ``i`` serves ``service_port + 1 + i``. vllm-router does not proxy the engines'
``/metrics``, so metric scrapes go to the engine ports.
"""

from typing import Optional, Tuple

# The address an engine binds: loopback on the host (local server). The docker runner sets 0.0.0.0 inside the
# container, where a published port reaches the container's address rather than its loopback, and publishes each
# engine port on the host's loopback only.
SUBMESH_ENGINE_HOST_ENV = "TT_SUBMESH_ENGINE_HOST"
DEFAULT_SUBMESH_ENGINE_HOST = "127.0.0.1"


def vllm_arg(vllm_args: dict, name: str, default=None):
    """A vllm_args value under either spelling (data_parallel_size or data-parallel-size)."""
    return vllm_args.get(name, vllm_args.get(name.replace("_", "-"), default))


def submesh_engine_count(vllm_args: dict) -> Optional[int]:
    """The number of submesh engines, data_parallel_size, when ``vllm_args`` set data_parallel_external_lb; else
    None (one server)."""
    if not vllm_arg(vllm_args, "data_parallel_external_lb"):
        return None
    data_parallel_size = int(vllm_arg(vllm_args, "data_parallel_size", 1))
    if data_parallel_size < 2:
        raise ValueError(
            "data_parallel_external_lb runs one engine per data-parallel rank: data_parallel_size must be at "
            f"least 2, got {data_parallel_size}"
        )
    return data_parallel_size


def submesh_engine_port(service_port, index: int) -> int:
    """Engine ``index``'s port."""
    return int(service_port) + 1 + index


def submesh_engine_metrics_urls(
    vllm_args: dict, service_port, host: str = DEFAULT_SUBMESH_ENGINE_HOST
) -> Tuple[str, ...]:
    """Each submesh engine's ``/metrics`` URL on ``host``; empty for one server."""
    engines = submesh_engine_count(vllm_args) or 0
    return tuple(
        f"http://{host}:{submesh_engine_port(service_port, index)}/metrics"
        for index in range(engines)
    )
