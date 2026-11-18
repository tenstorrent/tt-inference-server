# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Runner for agentic evals executed through the Harbor CLI.

Drives every agentic benchmark we run -- terminal-bench, tau3-bench, and
SWE-bench -- by shelling out to ``harbor run``. Harbor owns task acquisition,
sandboxing (docker / kubernetes / ...), the agent, and scoring, and writes a
``result.json`` the agentic parser reads directly, so this module only has to
translate an eval config into a Harbor job config.
"""

from __future__ import annotations

import copy
import functools
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from llm_module.agentic.local_datasets import local_task_pattern
from llm_module.agentic.progress import (
    TIMEOUT_EXIT_CODE,
    make_terminal_bench_probe,
    run_with_progress,
)

logger = logging.getLogger(__name__)

# Conservative fallback per-trial agent budget when the config leaves
# ``agent_timeout_sec`` unset (Harbor then uses each task's own default, which
# we cannot see from here). Only feeds the deadline math; the stall watchdog is
# the real protection, so err generous to avoid false kills.
_DEFAULT_AGENT_TIMEOUT_SEC = 60 * 60
_OPENAI_ENDPOINT_ENV = ("OPENAI_BASE_URL", "OPENAI_API_BASE")
# Inside a task container host loopback is not the inference server. A local
# endpoint is rewritten to the address containers reach the host at
# (``_docker_host_address``): Docker Desktop's ``host.docker.internal``, else the
# docker bridge gateway, which needs no ``extra_hosts`` entry (Docker rejects one
# on a service sharing the egress-control sidecar's network namespace).
# AGENTIC_DOCKER_HOST_ADDRESS overrides it.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DOCKER_HOST_NAME = "host.docker.internal"
_DOCKER_HOST_ADDRESS_ENV = "AGENTIC_DOCKER_HOST_ADDRESS"
_DEFAULT_BRIDGE_GATEWAY = "172.17.0.1"

# Harbor's built-in agent name for mini-swe-agent. When this agent is used we
# bring its generated model config to parity with the standalone SWE-bench
# harness (see ``_apply_mini_swe_agent_defaults``).
_MINI_SWE_AGENT = "mini-swe-agent"
# Custom agents that run inside the task container, as mini-swe-agent does
# (terminus-2 and other agents run on the host): the tau3 adapter's LLM agent
# runs run_tau3_llm_agent.py in the container.
_IN_CONTAINER_AGENT_IMPORTS = ("adapters.tau3-bench.tau3_llm_agent:Tau3LLMAgent",)


@dataclass(frozen=True)
class HarborRunConfig:
    task_name: str
    dataset: str
    agent: str
    model_name: str
    jobs_dir: Path
    api_base: str
    n_concurrent_trials: int
    n_attempts: int
    environment_type: str
    agent_kwargs: dict[str, Any]
    n_tasks: Optional[int]
    override_cpus: Optional[int]
    override_memory_mb: Optional[int]
    timeout_multiplier: Optional[float]
    agent_timeout_sec: Optional[float]
    agent_setup_timeout_multiplier: Optional[float] = None
    # Harbor's ResourceMode for the task's CPU and memory budget (``limit``,
    # ``request``, ``guarantee``, ...). ``None`` leaves Harbor's ``auto``.
    cpu_enforcement_policy: Optional[str] = None
    memory_enforcement_policy: Optional[str] = None
    task_names: list[str] = field(default_factory=list)
    exclude_task_names: list[str] = field(default_factory=list)
    # Rich Live progress: quiet=True shows only the loading bar; quiet=False
    # adds per-trial stage spinners (env start, agent start, verification).
    # In CI (non-TTY) Rich degrades Live to static line-by-line output, which
    # is exactly the visibility we want — each trial stage prints a line.
    quiet: bool = False
    yes: bool = True
    debug: bool = False
    agent_import_path: Optional[str] = None
    # Per-agent overrides. OpenAI endpoint variables are filled from api_base
    # unless explicitly set here, so stale host endpoints cannot take priority.
    agent_env: dict[str, str] = field(default_factory=dict)
    dataset_ref: Optional[str] = None
    environment_env: dict[str, str] = field(default_factory=dict)
    verifier_env: dict[str, str] = field(default_factory=dict)
    # Provider-specific environment knobs (namespace, image_mode, node_selector,
    # ... for the ``kubernetes`` environment). Only expressible through the
    # config file, so a non-empty value forces that path.
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    # Per-request LLM read timeout, injected into a mini-swe-agent run as
    # ``model.model_kwargs.timeout`` (see ``_apply_mini_swe_agent_defaults``).
    # litellm's OpenAI-compatible path otherwise defaults to an infinite read
    # timeout, so a never-answered request hangs the trial. ``None`` opts out.
    # Ignored for every non-mini agent (terminus-2, tau3, ... carry their own
    # timeout knob in ``agent_kwargs``).
    llm_timeout_sec: Optional[int] = 10 * 60
    # Allowance for Harbor's additive non-agent phases (environment build,
    # agent setup, verifier). Added to the agent timeout for each trial wave.
    per_task_overhead_sec: int = 20 * 60
    startup_grace_sec: int = 10 * 60
    stall_grace_sec: int = 5 * 60
    progress_log_interval_sec: int = 60
    # When False, heuristic wave/stall deadlines are log-only. Explicit hard
    # timeouts remain enforced.
    enforce_agent_deadline: bool = False
    # Interpreter whose bin/ holds the ``harbor`` CLI. When ``None`` the current
    # interpreter is used (standalone ``run_agentic.py`` already re-execs into
    # the EVALS_AGENTIC venv). Set on the release path, where the harness runs
    # as a child of the WORKFLOW_RUN_SCRIPT engine and must reach harbor explicitly.
    venv_python: Optional[Path] = None
    harbor_timeout_sec: Optional[float] = None
    # Locally generated copy of ``dataset`` (see ``local_datasets``). When set,
    # Harbor runs these task directories instead of resolving ``dataset`` in
    # the registry.
    dataset_path: Optional[Path] = None
    # Add the inference endpoint's host to Harbor's run-specific agent
    # allowlist, for tasks that block agent egress
    # (``[agent] network_mode = "no-network"``, e.g. DeepSWE).
    allow_agent_endpoint: bool = False
    # Pulled before ``harbor run`` on the docker environment (see
    # ``prepull_images``).
    prepull_images: list[str] = field(default_factory=list)
    prepull_attempts: int = 3
    prepull_timeout_sec: float = 30 * 60


def _apply_mini_swe_agent_defaults(
    agent_kwargs: dict[str, Any], config: HarborRunConfig
) -> None:
    """Bring a mini-swe-agent run to parity with the standalone SWE-bench harness.

    Harbor's ``mini-swe-agent`` agent takes an inline ``config`` mapping that it
    dumps to a mini-swe-agent YAML and injects with ``-c``. The standalone
    harness (``llm_module/agentic/swebench.py`` before the unification) always
    wrote three model defaults that harbor does not add on its own; without them
    a mini run inside harbor diverges from the standalone eval:

    * ``model.model_kwargs.drop_params=True`` -- litellm drops params the server
      rejects instead of erroring the whole request (needed for e.g. top_k in
      ``extra_body``).
    * ``model.model_kwargs.timeout`` -- litellm's OpenAI-compatible path defaults
      to an infinite read timeout, so a never-answered request otherwise hangs
      the trial until the whole-agent budget. ``llm_timeout_sec=None`` opts out.
    * ``model.cost_tracking='ignore_errors'`` -- a cost-lookup miss must not
      abort the run.

    Each value is a ``setdefault``, so an explicit per-eval value in
    ``agent_kwargs['config']`` always wins. ``agent_kwargs`` is mutated in place;
    the caller passes a deep copy so the shared eval config is never touched.
    """
    cfg = agent_kwargs.setdefault("config", {})
    if not isinstance(cfg, dict):
        return
    model = cfg.setdefault("model", {})
    if not isinstance(model, dict):
        return
    model.setdefault("cost_tracking", "ignore_errors")
    model_kwargs = model.setdefault("model_kwargs", {})
    if not isinstance(model_kwargs, dict):
        return
    model_kwargs.setdefault("drop_params", True)
    if config.llm_timeout_sec is not None:
        model_kwargs.setdefault("timeout", config.llm_timeout_sec)


def _get_agent_kwargs(config: HarborRunConfig) -> dict[str, Any]:
    """Copy agent options without injecting connection arguments."""
    agent_kwargs = copy.deepcopy(dict(config.agent_kwargs))
    if config.agent == _MINI_SWE_AGENT and config.agent_import_path is None:
        _apply_mini_swe_agent_defaults(agent_kwargs, config)
    return agent_kwargs


def _get_agent_endpoint(config: HarborRunConfig) -> str:
    """Resolve explicit endpoint overrides without consulting the host env."""
    values = [
        config.agent_env[key] for key in _OPENAI_ENDPOINT_ENV if key in config.agent_env
    ]
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("Agent endpoint overrides must be non-empty strings")
    if len(set(values)) > 1:
        raise ValueError("Conflicting OpenAI endpoint overrides in agent env")
    return values[0] if values else config.api_base


def _agent_in_task_container(config: HarborRunConfig) -> bool:
    if config.agent_import_path is None:
        return config.agent == _MINI_SWE_AGENT
    return config.agent_import_path in _IN_CONTAINER_AGENT_IMPORTS


@functools.cache
def _docker_host_address() -> str:
    """The address task containers reach this host at (see _LOOPBACK_HOSTS)."""
    override = os.environ.get(_DOCKER_HOST_ADDRESS_ENV, "").strip()
    if override:
        return override
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{.OperatingSystem}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if "Docker Desktop" in info.stdout:
            return _DOCKER_HOST_NAME
        bridge = subprocess.run(
            [
                "docker",
                "network",
                "inspect",
                "bridge",
                "--format",
                "{{range .IPAM.Config}}{{.Gateway}} {{end}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        gateways = bridge.stdout.split() if bridge.returncode == 0 else []
        if gateways:
            return gateways[0]
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Could not ask docker for the host address: %s", exc)
    return _DEFAULT_BRIDGE_GATEWAY


def _task_side_url(config: HarborRunConfig, url: str) -> str:
    """``url`` as a task container on the docker environment reaches it."""
    parsed = urlsplit(url)
    if config.environment_type != "docker" or parsed.hostname not in _LOOPBACK_HOSTS:
        return url
    host = f"[{parsed.hostname}]" if parsed.hostname == "::1" else parsed.hostname
    return urlunsplit(
        parsed._replace(netloc=parsed.netloc.replace(host, _docker_host_address(), 1))
    )


def _task_side_env(config: HarborRunConfig, env: dict[str, str]) -> dict[str, str]:
    """``env`` with its OpenAI endpoint values as a task container reaches them."""
    return {
        key: _task_side_url(config, value) if key in _OPENAI_ENDPOINT_ENV else value
        for key, value in env.items()
    }


def _get_task_side_endpoint(config: HarborRunConfig) -> str:
    """The endpoint as the agent reaches it: from the task container when it runs
    there, else (on the host) unchanged."""
    endpoint = _get_agent_endpoint(config)
    if _agent_in_task_container(config):
        endpoint = _task_side_url(config, endpoint)
    return endpoint


def _get_agent_env(config: HarborRunConfig) -> dict[str, str]:
    env = dict(config.agent_env)
    endpoint = _get_task_side_endpoint(config)
    env.update({key: endpoint for key in _OPENAI_ENDPOINT_ENV})
    return env


def _format_kwarg(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _write_harbor_config(config: HarborRunConfig) -> Path:
    config_path = config.jobs_dir / f"{config.task_name}_harbor_config.json"
    config.jobs_dir.mkdir(parents=True, exist_ok=True)

    task_names = config.task_names
    exclude_task_names = config.exclude_task_names
    if config.dataset_path is not None:
        # A locally generated copy has no registry version, so dataset_ref
        # does not apply.
        if config.dataset_ref:
            logger.warning(
                "Ignoring dataset_ref %r for %s: running the local copy at %s",
                config.dataset_ref,
                config.dataset,
                config.dataset_path,
            )
        dataset_config: dict[str, Any] = {"path": str(config.dataset_path)}
        task_names = [local_task_pattern(config.dataset, p) for p in task_names]
        exclude_task_names = [
            local_task_pattern(config.dataset, p) for p in exclude_task_names
        ]
    else:
        dataset_config = {"name": config.dataset}
        if config.dataset_ref:
            dataset_config["ref"] = config.dataset_ref
    if config.n_tasks is not None:
        dataset_config["n_tasks"] = config.n_tasks
    if task_names:
        dataset_config["task_names"] = task_names
    if exclude_task_names:
        dataset_config["exclude_task_names"] = exclude_task_names

    environment_config: dict[str, Any] = {"type": config.environment_type}
    if config.override_cpus is not None:
        environment_config["override_cpus"] = config.override_cpus
    if config.override_memory_mb is not None:
        environment_config["override_memory_mb"] = config.override_memory_mb
    if config.cpu_enforcement_policy is not None:
        environment_config["cpu_enforcement_policy"] = config.cpu_enforcement_policy
    if config.memory_enforcement_policy is not None:
        environment_config["memory_enforcement_policy"] = (
            config.memory_enforcement_policy
        )
    if config.environment_kwargs:
        environment_config["kwargs"] = config.environment_kwargs

    agent_config: dict[str, Any] = {
        "model_name": config.model_name,
        "override_timeout_sec": config.agent_timeout_sec,
        "kwargs": _get_agent_kwargs(config),
    }
    if config.agent_import_path:
        agent_config["import_path"] = config.agent_import_path
    else:
        agent_config["name"] = config.agent
    agent_env = _get_agent_env(config)
    if agent_env:
        agent_config["env"] = agent_env
    if config.allow_agent_endpoint:
        host = urlsplit(_get_task_side_endpoint(config)).hostname
        if not host:
            raise ValueError("allow_agent_endpoint needs an endpoint with a host")
        agent_config["extra_allowed_hosts"] = [host]

    # Environment and verifier env is read inside task containers (tau3's user
    # simulator and NL-assertion verifier call the model with it).
    if config.environment_env:
        environment_config["env"] = _task_side_env(config, config.environment_env)

    verifier_config: dict[str, Any] = {}
    if config.verifier_env:
        verifier_config["env"] = _task_side_env(config, config.verifier_env)

    harbor_config: dict[str, Any] = {
        "job_name": config.task_name,
        "jobs_dir": str(config.jobs_dir),
        "n_attempts": config.n_attempts,
        "n_concurrent_trials": config.n_concurrent_trials,
        "quiet": config.quiet,
        "environment": environment_config,
        "agents": [agent_config],
        "datasets": [dataset_config],
    }
    if verifier_config:
        harbor_config["verifier"] = verifier_config
    if config.agent_setup_timeout_multiplier is not None:
        harbor_config["agent_setup_timeout_multiplier"] = (
            config.agent_setup_timeout_multiplier
        )
    if config.timeout_multiplier is not None:
        harbor_config["timeout_multiplier"] = config.timeout_multiplier
    if config.agent_timeout_sec is not None:
        # Keep the agent override exact even if timeout_multiplier is used for
        # verifier or environment timeouts.
        harbor_config["agent_timeout_multiplier"] = 1.0

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(harbor_config, f, indent=2)

    return config_path


def _needs_config_file(config: HarborRunConfig) -> bool:
    return (
        _get_task_side_endpoint(config) != _get_agent_endpoint(config)
        or config.dataset_path is not None
        or config.allow_agent_endpoint
        or config.agent_timeout_sec is not None
        or config.agent_setup_timeout_multiplier is not None
        or config.agent_import_path is not None
        or bool(config.agent_env)
        or bool(config.environment_env)
        or bool(config.verifier_env)
        or bool(config.environment_kwargs)
        # The Hub ref is only expressible in the config file's datasets block.
        or bool(config.dataset_ref)
    )


def _annotate_result_file(result_file: Path) -> None:
    try:
        with result_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "_result_format" not in data:
            data["_result_format"] = "harbor"
            with result_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
    except (json.JSONDecodeError, IOError) as e:
        msg = (
            f"Could not annotate result file {result_file} with '_result_format' field "
            f"required for report processing: {e}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from e


def harbor_python(config: HarborRunConfig) -> Path:
    """The interpreter whose bin/ holds the harbor CLI for *config*."""
    return Path(config.venv_python or sys.executable)


def prepull_images(
    images: list[str],
    attempts: int,
    timeout_sec: float,
    retry_delay_sec: float = 30.0,
) -> list[str]:
    """``docker pull`` each image not already local, up to ``attempts`` times.

    A trial's environment build pulls its base image itself, under the task's
    build timeout and once per concurrent trial; on a slow or resetting registry
    link every build of the wave then fails together (tau3-bench on the Galaxy
    runner: ``python:3.12-slim`` at ~11 KB/s, reset mid-layer). One pull ahead
    of the run, retried, leaves the builds a local image (self-hosted runners
    keep it for later runs). A pull that still fails is logged and the run goes
    ahead: Harbor reports the build failure per trial. Returns the images that
    could not be pulled.
    """
    failed = []
    for image in images:
        try:
            present = (
                subprocess.run(
                    ["docker", "image", "inspect", image],
                    capture_output=True,
                    timeout=60,
                ).returncode
                == 0
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning(
                "Cannot inspect docker image %s (%s); not pre-pulling", image, e
            )
            failed.append(image)
            continue
        if present:
            logger.info("Docker image %s is already local", image)
            continue
        pulled = False
        for attempt in range(1, attempts + 1):
            logger.info(
                "Pre-pulling docker image %s (attempt %d/%d)", image, attempt, attempts
            )
            try:
                pulled = (
                    subprocess.run(
                        ["docker", "pull", image], timeout=timeout_sec
                    ).returncode
                    == 0
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "docker pull %s timed out after %.0f s", image, timeout_sec
                )
            except OSError as e:
                logger.warning("docker pull %s failed to start: %s", image, e)
                break
            if pulled:
                break
            if attempt < attempts:
                time.sleep(retry_delay_sec)
        if not pulled:
            logger.error(
                "Could not pull docker image %s in %d attempt(s); trials that build on it may fail",
                image,
                attempts,
            )
            failed.append(image)
    return failed


def run(config: HarborRunConfig) -> int:
    harbor_exec = harbor_python(config).parent / "harbor"

    if config.prepull_images and config.environment_type == "docker":
        prepull_images(
            config.prepull_images, config.prepull_attempts, config.prepull_timeout_sec
        )

    if _needs_config_file(config):
        harbor_config_path = _write_harbor_config(config)
        cmd = [str(harbor_exec), "run", "--config", str(harbor_config_path)]
        if config.yes:
            cmd.append("--yes")
    else:
        config.jobs_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(harbor_exec),
            "run",
            "--dataset",
            config.dataset,
            "--agent",
            config.agent,
            "--model",
            config.model_name,
            "--jobs-dir",
            str(config.jobs_dir),
            "--job-name",
            config.task_name,
            "--n-concurrent",
            str(config.n_concurrent_trials),
            "--n-attempts",
            str(config.n_attempts),
            "--env",
            config.environment_type,
        ]
        if config.quiet:
            cmd.append("--quiet")
        if config.yes:
            cmd.append("--yes")
        if config.n_tasks is not None:
            cmd.extend(["--n-tasks", str(config.n_tasks)])
        if config.override_cpus is not None:
            cmd.extend(["--override-cpus", str(config.override_cpus)])
        if config.override_memory_mb is not None:
            cmd.extend(["--override-memory-mb", str(config.override_memory_mb)])
        if config.cpu_enforcement_policy is not None:
            cmd.extend(["--cpus", config.cpu_enforcement_policy])
        if config.memory_enforcement_policy is not None:
            cmd.extend(["--memory", config.memory_enforcement_policy])
        if config.timeout_multiplier is not None:
            cmd.extend(["--timeout-multiplier", str(config.timeout_multiplier)])
        if config.agent_setup_timeout_multiplier is not None:
            cmd.extend(
                [
                    "--agent-setup-timeout-multiplier",
                    str(config.agent_setup_timeout_multiplier),
                ]
            )
        for task_name in config.task_names:
            cmd.extend(["--include-task-name", task_name])
        for task_name in config.exclude_task_names:
            cmd.extend(["--exclude-task-name", task_name])

        agent_kwargs = _get_agent_kwargs(config)
        for key, value in agent_kwargs.items():
            cmd.extend(["--agent-kwarg", f"{key}={_format_kwarg(value)}"])
        for key, value in _get_agent_env(config).items():
            cmd.extend(["--agent-env", f"{key}={value}"])

    if config.debug:
        cmd.append("--debug")

    job_dir = config.jobs_dir / config.task_name
    agent_timeout = (
        config.agent_timeout_sec
        if config.agent_timeout_sec is not None
        else _DEFAULT_AGENT_TIMEOUT_SEC
    )
    per_task_budget = agent_timeout + config.per_task_overhead_sec
    # ``harbor_timeout_sec`` is an optional flat backstop kept from the unified
    # harness; the wave-aware stall/ceiling watchdog is the primary protection.
    # Host-executed agents use this endpoint directly. Container-executed
    # mini-swe receives its translated endpoint through the agent config.
    process_env = os.environ.copy()
    endpoint = _get_agent_endpoint(config)
    process_env.update({key: endpoint for key in _OPENAI_ENDPOINT_ENV})
    rc = run_with_progress(
        cmd,
        cwd=None,
        env=process_env,
        probe=make_terminal_bench_probe(job_dir),
        label=config.task_name,
        per_task_budget_s=per_task_budget,
        concurrency=config.n_concurrent_trials,
        startup_grace_s=config.startup_grace_sec,
        stall_grace_s=config.stall_grace_sec,
        log_interval_s=config.progress_log_interval_sec,
        hard_timeout_s=config.harbor_timeout_sec,
        enforce_deadlines=config.enforce_agent_deadline,
        log=logger,
    )
    # A watchdog timeout can still leave useful per-trial diagnostics in
    # result.json. Annotate that file, but preserve rc=124: Harbor computes
    # accuracy over completed trials, so treating a partial file as success can
    # inflate its score by excluding unfinished trials from the denominator.
    if rc != 0 and rc != TIMEOUT_EXIT_CODE:
        return rc
    result_path = job_dir / "result.json"
    if rc == TIMEOUT_EXIT_CODE and not result_path.exists():
        logger.error(
            "Harbor timed out before writing any results; nothing to annotate."
        )
        return rc
    if rc == TIMEOUT_EXIT_CODE:
        logger.warning(
            "Harbor hit the deadline; annotating the partial results in %s.",
            result_path,
        )
    try:
        _annotate_result_file(result_path)
    except RuntimeError:
        if rc != TIMEOUT_EXIT_CODE:
            raise
        logger.warning(
            "Harbor timed out while result.json was incomplete; preserving rc=%d.",
            rc,
            exc_info=True,
        )
    return rc
