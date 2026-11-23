# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""``run.py --cohost-with``: serve several vLLM models on the same chips, from one container.

A Tenstorrent chip can be opened by one process at a time, so the models share a process
(tt-cohost) inside one container. ``--model`` is served on ``--service-port``, each
``--cohost-with`` model on the next port, and the group's status endpoint on the port after
the last member. This module resolves and checks the members and writes the group file the
container reads (``run_vllm_api_server.py --cohost-group``).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from workflows.workflow_types import InferenceEngine

logger = logging.getLogger("run_log")

COHOST_GROUP_VERSION = 1
CONTAINER_GROUP_DIR = Path("/home/container_app_user/model_specs")


@dataclass
class CohostMember:
    name: str
    port: int
    model_spec: object


def parse_cohost_with(value: str) -> List[str]:
    """``"a, b"`` -> ``["a", "b"]``; empty names and repeats are refused."""
    names = [name.strip() for name in value.split(",")]
    if not all(names):
        raise ValueError(
            f"--cohost-with must be a comma-separated list of models (got {value!r})."
        )
    if len(set(names)) != len(names):
        raise ValueError(f"--cohost-with names a model twice (got {value!r}).")
    return names


def validate_cohost_members(
    members: List[CohostMember], override_docker_image: Optional[str] = None
) -> None:
    """Refuse a group that cannot share one process, naming the member and why."""
    names = [m.name for m in members]
    if len(set(names)) != len(names):
        raise ValueError(f"a co-host group serves each model once (got {names}).")
    first = members[0]
    for m in members:
        spec = m.model_spec
        if spec.inference_engine != InferenceEngine.VLLM.value:
            raise ValueError(
                f"{m.name} is served by {spec.inference_engine}; only vLLM models can "
                "share chips through --cohost-with. Serve it on separate chips."
            )
        if spec.impl.impl_id == "quetzal":
            raise ValueError(
                f"{m.name} is a Quetzal package; it cannot join a co-host group. "
                "Serve it on separate chips."
            )
        if (
            not override_docker_image
            and spec.docker_image != first.model_spec.docker_image
        ):
            raise ValueError(
                f"{first.name} runs on {first.model_spec.docker_image} and {m.name} on "
                f"{spec.docker_image}. Models sharing chips share one process, so they "
                "need the same image (one tt-metal build). Serve them on separate chips, "
                "or pass --override-docker-image with an image both run on."
            )


def resolve_cohost_members(
    model_spec, runtime_config, names: List[str], resolve: Callable
) -> List[CohostMember]:
    """``--model`` plus each ``--cohost-with`` model, resolved on the same device.

    ``resolve(model=, device=)`` returns ``(model_spec, impl, engine)`` like
    ``get_runtime_model_spec``. Ports follow ``--service-port``.
    """
    port = int(runtime_config.service_port)
    members = [CohostMember(model_spec.model_name, port, model_spec)]
    for offset, name in enumerate(names, start=1):
        spec, _impl, _engine = resolve(model=name, device=runtime_config.device)
        if runtime_config.override_docker_image:
            object.__setattr__(
                spec, "docker_image", runtime_config.override_docker_image
            )
        members.append(CohostMember(spec.model_name, port + offset, spec))
    validate_cohost_members(members, runtime_config.override_docker_image)
    return members


def _member_spec(model_spec) -> dict:
    spec = model_spec.get_serialized_dict()
    # This run's CLI arguments (which can hold secrets) are not the container's business.
    spec.pop("cli_args", None)
    return spec


def cohost_group_document(members: List[CohostMember], device: str) -> dict:
    return {
        "cohost_group_version": COHOST_GROUP_VERSION,
        "device": device,
        "status_port": members[-1].port + 1,
        "members": [
            {
                "name": m.name,
                "port": m.port,
                "runtime_model_spec": _member_spec(m.model_spec),
            }
            for m in members
        ],
    }


def write_cohost_group(
    model_spec, runtime_config, output_dir, resolve: Callable
) -> Path:
    """Resolve and check the group, write its file, and return the file's path."""
    names = parse_cohost_with(runtime_config.cohost_with)
    members = resolve_cohost_members(model_spec, runtime_config, names, resolve)
    document = cohost_group_document(members, runtime_config.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"cohost_group_{runtime_config.run_id}.json"
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False))
    # Bind-mounted into the container, which reads it as the image's user.
    path.chmod(0o644)
    summary = ", ".join(f"{m.name} on port {m.port}" for m in members)
    logger.info(
        f"Co-host group: {summary}; status on port {document['status_port']} ({path})"
    )
    return path


def cohost_group_ports(group_path) -> List[int]:
    """Every port the group's container serves: each member's, then the status port."""
    document = json.loads(Path(group_path).read_text())
    return [int(m["port"]) for m in document["members"]] + [
        int(document["status_port"])
    ]


def cohost_docker_args(
    group_path, bind_host: str, repo_root: Path, dev_mode: bool
) -> Tuple[List[str], List[str]]:
    """``(docker run flags, server args)`` that serve the group in the vLLM image.

    The first member's port is published by the normal path. The repo's in-container
    launcher is mounted (as ``--dev-mode`` does) so images published before the
    ``--cohost-group`` option can serve a group too.
    """
    group_path = Path(group_path)
    container_group_path = CONTAINER_GROUP_DIR / group_path.name
    flags = []
    for port in cohost_group_ports(group_path)[1:]:
        flags += ["--publish", f"{bind_host}:{port}:{port}"]
    flags += [
        "--mount",
        f"type=bind,src={group_path},dst={container_group_path},readonly",
    ]
    if not dev_mode:
        app = "/home/container_app_user/app"
        flags += [
            "--mount",
            f"type=bind,src={repo_root}/vllm-tt-metal/src,dst={app}/src",
            "--mount",
            f"type=bind,src={repo_root}/utils,dst={app}/utils",
        ]
    return flags, ["--cohost-group", str(container_group_path)]
