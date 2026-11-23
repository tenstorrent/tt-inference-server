# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Serve a co-host group: several vLLM models in one process on the same chips.

A Tenstorrent chip can be opened by one process at a time, so models that share chips
have to share a process. tt-cohost is that process: it opens one mesh, boots every member
in phases (load all, compile all, then capture traces member by member) and gives each
member its own stock vLLM OpenAI server on its own port. This module turns the group file
written by ``run.py --cohost-with`` into a tt-cohost config and starts tt-cohost.

Each member gets the environment and ``vllm serve`` arguments its own server would get:
the setup steps of ``run_vllm_api_server.main`` run once per member, from the same
starting environment, and whatever they set becomes that member's ``env`` (tt-cohost
applies it around the member's load and in its vLLM frontend).
"""

import argparse
import importlib
import importlib.util
import json
import logging
import os
import runpy
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

GROUP_VERSION = 1
COHOST_CONFIG_VERSION = 1
TT_COHOST_LAUNCH_MODULE = "tt_cohost.launch"
# Installed at container start when the image does not ship tt-cohost.
TT_COHOST_SHA = "7211ac7343e5ba9c33c530f2ad09095e37c4c3cb"
DEFAULT_TT_COHOST_PIP_SPEC = (
    "tt-cohost @ https://github.com/jashansinghTT/tt-cohost/archive/"
    f"{TT_COHOST_SHA}.tar.gz"
)
# The keys of the plugin's tt config that configure the mesh. The group opens one mesh,
# so its members must agree on them; every other tt key stays per member.
MESH_KEYS = (
    "fabric_config",
    "dispatch_core_axis",
    "worker_l1_size",
    "l1_small_size",
)
# Every member captures its traces into the one trace region, so the group's is the sum.
# A member that sets none gets the plugin's default.
DEFAULT_TRACE_REGION_SIZE = 50_000_000
# Per-model paths run.py passes for --model. They belong to the first member only.
FIRST_MEMBER_ENV = ("MODEL_WEIGHTS_DIR", "TT_CACHE_PATH")


def load_group(path) -> dict:
    group = json.loads(Path(path).read_text())
    version = group.get("cohost_group_version")
    if version != GROUP_VERSION:
        raise ValueError(
            f"co-host group file {path} has version {version!r}; this server reads {GROUP_VERSION}"
        )
    if len(group.get("members") or []) < 2:
        raise ValueError(f"co-host group file {path} names fewer than two members")
    return group


@contextmanager
def _restored_process_state():
    """Let a member's setup set env vars and sys.argv, then put both back."""
    env, argv = dict(os.environ), list(sys.argv)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(env)
        sys.argv = argv


def member_setup(server, spec: dict, port: int, *, no_auth: bool, first: bool):
    """The env vars and ``vllm serve`` argv ``server.main()`` would set for this model.

    Returns ``(env, argv)``: ``env`` holds only what the setup added or changed, and
    nothing stays set in this process afterwards.
    """
    with _restored_process_state():
        host_paths = {k: os.environ.pop(k) for k in FIRST_MEMBER_ENV if k in os.environ}
        start = dict(os.environ)
        if first:
            os.environ.update(host_paths)

        device_type = server.normalize_device_type(spec["device_type"])
        if not os.getenv("TT_CACHE_PATH"):
            server.set_cache_paths(spec, device_type)
        if not os.getenv("MODEL_WEIGHTS_DIR") and not os.getenv("MULTIHOST_ROLE"):
            server.ensure_weights_available(spec)
        server.register_tt_models(spec.get("impl", {}).get("impl_id"))
        server.set_runtime_env_vars(spec)
        server.set_metal_timeout_env_vars()
        server.runtime_settings(spec, no_auth=no_auth)

        vllm_args = dict(spec["device_model_spec"]["vllm_args"])
        vllm_args["port"] = str(port)
        server.absorb_plugin_config_into_additional_config(vllm_args)
        server.inject_tt_data_parallel(vllm_args)
        server.inject_tt_weight_bridge_dir(vllm_args)
        server.inject_tt_colocated_fabric_config(vllm_args)
        server.set_vllm_sys_argv(argparse.Namespace(service_port=None), [], vllm_args)
        argv = list(sys.argv[1:])

        env = {k: v for k, v in os.environ.items() if start.get(k) != v}
        dropped = sorted(set(start) - set(os.environ))
    if dropped:
        logger.warning(
            f"{spec['model_name']}: its setup unset {dropped}; a co-host member can only add "
            "env vars, so they stay set for it"
        )
    return env, argv


def split_vllm_argv(server, argv: list) -> tuple:
    """``(model, args, tt_config)`` from a ``vllm serve`` argv.

    tt-cohost passes ``--model``, ``--host`` and ``--port`` itself, so they leave ``args``.
    """
    model, args, tt_config = None, [], {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            # The further values of a multi-value flag (e.g. --served-model-name a b).
            args.append(token)
            index += 1
            continue
        name = server._normalize_vllm_arg_name(token)
        if "=" in token:
            value, step = token.split("=", 1)[1], 1
        elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            value, step = argv[index + 1], 2
        else:
            value, step = None, 1
        if name == "model":
            model = value
        elif name not in ("host", "port"):
            if name == "additional_config" and value:
                tt_config = json.loads(value).get("tt") or {}
            args.extend(argv[index : index + step])
        index += step
    if not model:
        raise ValueError(f"no --model in the vLLM arguments {argv}")
    return model, args, tt_config


def group_tt_config(tt_configs: dict) -> dict:
    """The tt config the group's one mesh is opened with, from each member's own."""
    merged, owner = {}, {}
    trace_region_size = 0
    for name, tt_config in tt_configs.items():
        trace_region_size += int(
            tt_config.get("trace_region_size", DEFAULT_TRACE_REGION_SIZE)
        )
        for key in MESH_KEYS:
            if key not in tt_config:
                continue
            if key in merged and merged[key] != tt_config[key]:
                raise ValueError(
                    f"{owner[key]} and {name} need different {key} ({merged[key]!r} vs "
                    f"{tt_config[key]!r}). A co-host group opens one mesh: serve them on "
                    "separate chips."
                )
            merged[key], owner[key] = tt_config[key], name
    merged["trace_region_size"] = trace_region_size
    return merged


def build_cohost_config(server, group: dict, *, no_auth: bool = False) -> tuple:
    """``(config, shared_env)`` that serve every member of ``group`` with tt-cohost.

    ``config`` is a tt-cohost config (version 1). ``shared_env`` holds the env vars every
    member sets to the same value (MESH_DEVICE, the op timeout, ...): they belong to the
    whole process, which opens the mesh, so they leave the members' own ``env``.
    """
    # Importing vLLM and its model registry (torch, OpenMP) sets process-wide env vars: do
    # it first, so they stay with the process instead of landing in the first member's env.
    getattr(importlib.import_module("vllm"), "ModelRegistry", None)
    members, tt_configs = [], {}
    for index, member in enumerate(group["members"]):
        name, port = member["name"], int(member["port"])
        env, argv = member_setup(
            server,
            member["runtime_model_spec"],
            port,
            no_auth=no_auth,
            first=index == 0,
        )
        model, args, tt_configs[name] = split_vllm_argv(server, argv)
        members.append(
            {
                "name": name,
                "kind": "vllm",
                "port": port,
                "model": model,
                "args": args,
                "env": env,
            }
        )
    mesh_devices = {
        m["name"]: m["env"].get("MESH_DEVICE") or os.getenv("MESH_DEVICE")
        for m in members
    }
    if len(set(mesh_devices.values())) != 1:
        raise ValueError(f"the members need different meshes: {mesh_devices}")
    shared_env = {
        key: value
        for key, value in members[0]["env"].items()
        if all(m["env"].get(key) == value for m in members[1:])
    }
    for m in members:
        m["env"] = {k: v for k, v in m["env"].items() if k not in shared_env}
    config = {
        "cohost_config_version": COHOST_CONFIG_VERSION,
        "mesh_device": next(iter(mesh_devices.values())),
        "tt_config": group_tt_config(tt_configs),
        "members": members,
    }
    if group.get("status_port"):
        config["status_port"] = int(group["status_port"])
    return config, shared_env


def ensure_tt_cohost() -> None:
    """Install tt-cohost into a private directory if this image does not ship it."""
    if importlib.util.find_spec("tt_cohost") is not None:
        return
    pip_spec = os.getenv("TT_COHOST_PIP_SPEC", DEFAULT_TT_COHOST_PIP_SPEC)
    target = Path(os.getenv("TT_COHOST_SITE_DIR", "/tmp/tt-cohost-site"))
    logger.info(f"tt-cohost is not in this image; installing {pip_spec} into {target}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps"]
        + ["--target", str(target), pip_spec],
        check=True,
    )
    sys.path.insert(0, str(target))
    # The members' vLLM frontends are child processes: they need it on PYTHONPATH.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(target), os.environ.get("PYTHONPATH")) if p
    )
    importlib.invalidate_caches()


def run_cohost_group(server, group_path, *, no_auth: bool = False) -> None:
    """Boot every member of the group in this process; returns only when the group stops."""
    group = load_group(group_path)
    names = [m["name"] for m in group["members"]]
    logger.info(f"Serving co-host group {names} on {group.get('device')}")
    config, shared_env = build_cohost_config(server, group, no_auth=no_auth)
    for key in FIRST_MEMBER_ENV:
        os.environ.pop(key, None)  # now in the first member's env
    os.environ.update(shared_env)
    logger.info(f"Env shared by every member: {sorted(shared_env)}")
    config_dir = Path(tempfile.mkdtemp(prefix="tt-cohost-config-"))
    config_path = config_dir / "cohost_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    logger.info(
        f"tt-cohost config written to {config_path}:\n{json.dumps(config, indent=2)}"
    )

    ensure_tt_cohost()
    # tt-cohost boots and warms up every member before it serves, so the solo server's
    # background trace-capture requests are not sent.
    sys.argv = [TT_COHOST_LAUNCH_MODULE, "--config", str(config_path)]
    runpy.run_module(TT_COHOST_LAUNCH_MODULE, run_name="__main__", alter_sys=True)
