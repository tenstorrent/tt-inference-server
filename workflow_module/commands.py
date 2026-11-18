# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    # Annotation-only imports (``from __future__ import annotations`` keeps them
    # unevaluated at runtime). Kept out of the import path so lightweight callers
    # — e.g. run.py constructing a ServerCommand — need not pull the heavy
    # test_module.context / report_module stack.
    from test_module import MediaContext

    from .execution import OrchestratorMetadata, WorkflowResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:
    command_name: str
    return_code: int
    error: Optional[str] = None
    payload: Optional[Any] = None

    @property
    def succeeded(self) -> bool:
        return self.return_code == 0


class Command(ABC):
    name: str = ""

    @abstractmethod
    def execute(self) -> CommandResult: ...


class ServerMode(str, Enum):
    """How :class:`ServerCommand` brings up the inference server."""

    DOCKER = "docker"
    LOCAL = "local"


@dataclass(frozen=True)
class ServerLaunchSpec:
    """Everything :class:`ServerCommand` needs to bring up an inference server.

    Carries the launcher-side objects (``model_spec``, ``runtime_config``,
    ``setup_config``) that :func:`workflows.run_docker_server.run_docker_server`
    and :func:`workflows.run_local_server.run_local_server` expect. They are
    typed ``Any`` here so the command model stays free of a hard import
    dependency on the launcher stack.

    ``mode`` selects the launcher (:class:`ServerMode`). ``json_fpath`` is the
    runtime model-spec JSON path the launchers persist / read (the docker
    launcher only forwards it in ``--dev-mode``).
    """

    mode: ServerMode
    model_spec: Any
    runtime_config: Any
    setup_config: Any
    json_fpath: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.mode, ServerMode):
            return
        try:
            object.__setattr__(self, "mode", ServerMode(self.mode))
        except ValueError as e:
            raise ValueError(f"unknown server mode: {self.mode!r}") from e


# The dispatch watchdog's signature. tt-metal raises this when the device stops
# draining its fetch queue, which on Galaxy has shown up intermittently during the
# model's prefill warmup sweep. It is worth naming explicitly so a retry says *why*
# it retried instead of reporting a generic timeout.
_HANG_MARKERS = (
    "potential hang detected",
    "device timeout in fetch queue wait",
    "Timeout detected",
)

_FATAL_STARTUP_MARKERS = (
    "Worker warmup failed; stopping all spawned workers",
    "Leader worker 0 failed warmup; aborting remaining worker startup",
    "Configured service reported fatal worker error",
)


def _server_boot_attempts() -> int:
    """How many times to try bringing the server up.

    Defaults to 2. Galaxy bring-up can hit an intermittent device stall during the
    model's prefill warmup sweep, and a single stall otherwise fails an entire
    release run. Set TT_SERVER_BOOT_ATTEMPTS=1 to restore strict fail-fast.

    A retry is only meaningful if bring-up waits long enough to observe whether the
    server actually came up, so this also governs the readiness wait below.
    """
    import os

    raw = os.getenv("TT_SERVER_BOOT_ATTEMPTS", "2")
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid TT_SERVER_BOOT_ATTEMPTS=%r; using 2", raw)
        return 2


# Time a first boot spends outside tensor-cache generation: weight download, load
# and server start. Matches the Helm chart's progressDeadlineSeconds
# (tensor_cache_timeout + 1800, workflows/helm_generator/base_mapper.py).
_BOOT_OVERHEAD_SECONDS = 1800.0
_DEFAULT_READY_TIMEOUT_SECONDS = 3600.0


def _spec_ready_timeout(model_spec: Any) -> Optional[float]:
    """The model's own first-boot budget: tensor_cache_timeout plus boot overhead."""
    device_model_spec = getattr(model_spec, "device_model_spec", None)
    raw = getattr(device_model_spec, "tensor_cache_timeout", None)
    try:
        tensor_cache_timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if tensor_cache_timeout <= 0:
        return None
    return tensor_cache_timeout + _BOOT_OVERHEAD_SECONDS


def _readiness_timeout(model_spec: Any = None) -> float:
    """How long bring-up waits for /health before calling the attempt failed.

    TT_SERVER_READY_TIMEOUT_SECONDS wins when set. Otherwise the model's own budget
    is used: its tensor_cache_timeout plus time for the weight download and server
    start, the same deadline the Helm chart gives the deployment. A cold 70B start on
    QB2 measured ~59 min (#5363), so a flat 1h cut it off, and the retry then lost
    the first attempt's progress. Genuine stalls are caught by the log hang markers
    long before this deadline, so this is a backstop, not the primary detector.
    """
    import os

    raw = os.getenv("TT_SERVER_READY_TIMEOUT_SECONDS")
    if raw is not None:
        try:
            return max(1.0, float(raw))
        except ValueError:
            logger.warning(
                "Invalid TT_SERVER_READY_TIMEOUT_SECONDS=%r; using the model's budget",
                raw,
            )
    return _spec_ready_timeout(model_spec) or _DEFAULT_READY_TIMEOUT_SECONDS


# Leading words of the readiness-wait reason when the deadline passes.
_TIMED_OUT = "timed out"

# vLLM images built before v0.19.0 (#4397) skip the weight download when the
# weights directory is non-empty, so a download cut off by a timeout is never
# resumed and the model fails on every later start until the volume is deleted.
_RESUMABLE_DOWNLOAD_MIN_VERSION = (0, 19, 0)


def _parse_version(version: Any) -> Optional[tuple]:
    import re

    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(version or ""))
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _container_cannot_resume_download(spec: ServerLaunchSpec) -> bool:
    """True when a timed-out boot may have left a half-downloaded weights dir behind.

    That happens only when the weights are downloaded inside the container by a
    vLLM image whose launcher predates resumable downloads. Retrying such a boot
    cannot succeed. The container does not download when run.py fetched the weights
    on the host (--host-volume, --host-hf-cache), or when it mounts existing weights
    (host_model_weights_mount_dir: --host-weights-dir, or MODEL_SOURCE=local with
    MODEL_WEIGHTS_DIR), because the container then gets MODEL_WEIGHTS_DIR.
    """
    if spec.mode is not ServerMode.DOCKER:
        return False
    model_spec = spec.model_spec
    if getattr(model_spec, "inference_engine", None) != "vLLM":
        return False
    version = _parse_version(getattr(model_spec, "version", None))
    if version is None or version >= _RESUMABLE_DOWNLOAD_MIN_VERSION:
        return False
    setup_config = spec.setup_config
    return not any(
        getattr(setup_config, name, None)
        for name in (
            "host_model_volume_root",
            "host_hf_cache",
            "host_weights_dir",
            "host_model_weights_mount_dir",
        )
    )


def _payload_get(payload: Any, key: str) -> Any:
    if isinstance(payload, Mapping):
        return payload.get(key)
    return None


def _server_log_path(payload: Any) -> Optional[Path]:
    """Path to the launched server's log, if it exists and is readable."""
    raw = _payload_get(payload, "local_log_file_path") or _payload_get(
        payload, "docker_log_file_path"
    )
    if not raw:
        return None
    path = Path(raw)
    return path if path.exists() else None


def _scan_log_for_markers(payload: Any, markers: Sequence[str]) -> Optional[str]:
    """Return the first marker from ``markers`` found in the server log, if any."""
    log_path = _server_log_path(payload)
    if not log_path:
        return None
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return None
    for marker in markers:
        if marker in text:
            return marker
    return None


def _scan_log_for_hang(payload: Any) -> Optional[str]:
    """Return the hang marker found in the server log, if any."""
    return _scan_log_for_markers(payload, _HANG_MARKERS)


def _scan_log_for_fatal_startup_error(payload: Any) -> Optional[str]:
    """Return a fatal worker/warmup failure marker found in the server log, if any."""
    return _scan_log_for_markers(payload, _FATAL_STARTUP_MARKERS)


def _fatal_health_error(exc: Any) -> Optional[str]:
    """Extract a terminal worker failure from an HTTP error response body, if present."""
    import json

    read = getattr(exc, "read", None)
    if not callable(read):
        return None
    try:
        raw = read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, Mapping):
        return None
    error = str(data.get("error") or "")
    if "is dead" in error or error == "no workers are alive":
        return error
    return None


def _server_is_alive(spec: ServerLaunchSpec, payload: Any) -> Optional[bool]:
    """Liveness of the launched server: True, False, or None when unobservable.

    Deliberately does not shell out. Both launchers retain their foreground process
    handle when one is available, so polling that handle observes an exited Docker
    container as well as an exited local server without adding a separate ``docker
    inspect`` failure mode. Older local launch payloads retain the PID fallback.
    """
    process = _payload_get(payload, "process")
    if process is not None:
        poll = getattr(process, "poll", None)
        if callable(poll):
            return poll() is None
    if spec.mode is ServerMode.LOCAL:
        pid = _payload_get(payload, "pid")
        if pid is None:
            return None
        return Path(f"/proc/{pid}").exists()
    return None


def _wait_until_ready(spec: ServerLaunchSpec, payload: Any) -> Optional[str]:
    """Block until the server answers /health.

    Returns None once ready, else a short reason string. The launchers return as soon
    as the process/container exists (a ~2s grace period), long before the model has
    finished loading and warming up -- so a device stall during warmup would otherwise
    be reported as a successful bring-up and only surface later as a confusing
    health-check failure in whatever workflow ran next.
    """
    import time
    import urllib.error
    import urllib.request

    # Readiness polling rides along with the retry opt-in. With a single attempt
    # there is nothing to do differently if the server is unhealthy, and polling
    # would add process/network calls to a path that deliberately has none.
    if _server_boot_attempts() <= 1:
        return None

    port = _payload_get(payload, "service_port") or getattr(
        spec.runtime_config, "service_port", None
    )
    if not port:
        # Unknown port (also the unit-test path, where runtime_config is a stub):
        # nothing to probe, so preserve the previous fire-and-forget behaviour.
        return None

    url = f"http://127.0.0.1:{port}/health"

    # Only wait on a server we can actually observe. The server log is what makes a
    # stalled bring-up diagnosable (and detectable, via the hang markers); if it does
    # not exist there is nothing here that was really started -- e.g. a stubbed
    # launcher -- so fall back to the previous fire-and-forget behaviour rather than
    # burning the whole timeout.
    if not _server_log_path(payload):
        logger.info(
            "No readable server log for this handle; skipping readiness wait for %s",
            url,
        )
        return None

    timeout = _readiness_timeout(spec.model_spec)
    deadline = time.time() + timeout
    logger.info(
        "Waiting up to %.0fs for inference server readiness at %s ...", timeout, url
    )

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    logger.info("Inference server is ready at %s", url)
                    return None
        except urllib.error.HTTPError as exc:
            fatal = _fatal_health_error(exc)
            if fatal:
                return f"server reported fatal health error ({fatal})"
        except (urllib.error.URLError, OSError, ValueError):
            pass

        if _server_is_alive(spec, payload) is False:
            marker = _scan_log_for_hang(payload)
            if marker:
                return f"server process exited during startup (device hang: {marker!r})"
            fatal_marker = _scan_log_for_fatal_startup_error(payload)
            if fatal_marker:
                return f"server process exited during startup ({fatal_marker!r})"
            return "server process exited during startup"

        # A hung device keeps the process alive but never serves traffic, so the
        # log marker is the only way to fail fast instead of burning the timeout.
        marker = _scan_log_for_hang(payload)
        if marker:
            return f"device hang detected during startup ({marker!r})"

        fatal_marker = _scan_log_for_fatal_startup_error(payload)
        if fatal_marker:
            return f"worker warmup failed during startup ({fatal_marker!r})"

        time.sleep(5)

    return f"{_TIMED_OUT} after {timeout:.0f}s waiting for {url}"


def _teardown_server(spec: ServerLaunchSpec, payload: Any) -> None:
    """Stop a half-started server so the next attempt gets the devices back."""
    import signal
    import subprocess
    import time

    try:
        if spec.mode is ServerMode.LOCAL:
            pid = _payload_get(payload, "pid")
            if pid:
                import os

                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(pid, sig)
                    except (ProcessLookupError, PermissionError, OSError):
                        break
                    time.sleep(5)
                    if not Path(f"/proc/{pid}").exists():
                        break
        else:
            container = _payload_get(payload, "container_name") or _payload_get(
                payload, "container_id"
            )
            if container:
                subprocess.run(
                    ["docker", "stop", container], capture_output=True, timeout=120
                )
    except Exception:  # teardown is best-effort; never mask the original failure
        logger.exception("Server teardown failed (continuing)")


class ServerCommand(Command):
    """Bring up the inference server as the first step of a run.

    Delegates to the registered :class:`ServerLifecycle` (vendor adapter:
    docker/local launchers) so server bring-up is a command in the same list
    the :class:`WorkflowRunner` executes.
    """

    name = "server"

    def __init__(self, launch: ServerLaunchSpec) -> None:
        self.launch = launch

    def execute(self) -> CommandResult:
        spec = self.launch
        attempts = _server_boot_attempts()
        last_error = None
        attempts_made = 0

        for attempt in range(1, attempts + 1):
            attempts_made = attempt
            if attempt > 1:
                logger.warning(
                    "Retrying server bring-up (attempt %d/%d) after: %s",
                    attempt,
                    attempts,
                    last_error,
                )
            try:
                payload = self._launch_once(spec)
            # launcher itself failed (container/process died at once)
            except Exception as e:
                logger.exception("Server bring-up failed: %s", e)
                last_error = str(e)
                _teardown_server(spec, None)
                continue

            ready_error = _wait_until_ready(spec, payload)
            if ready_error is None:
                if attempt > 1:
                    # Distinctive, greppable line: a green run that needed a retry is
                    # still hiding an intermittent bring-up failure, and that must stay
                    # visible in CI rather than being silently absorbed.
                    logger.warning(
                        "TT_SERVER_BOOT_RETRY_SUCCEEDED attempt=%d/%d prior_error=%s",
                        attempt,
                        attempts,
                        last_error,
                    )
                return CommandResult(
                    command_name=self.name, return_code=0, payload=payload
                )

            last_error = ready_error
            logger.error("Server did not become ready: %s", ready_error)
            _teardown_server(spec, payload)

            if ready_error.startswith(_TIMED_OUT) and _container_cannot_resume_download(
                spec
            ):
                logger.error(
                    "Not retrying: image v%s cannot resume a weight download that the "
                    "timeout may have cut off, so a retry would fail on incomplete "
                    "weights. Re-run with --host-volume (weights are downloaded on the "
                    "host and resumed), or delete this model's Docker volume first. "
                    "Raise TT_SERVER_READY_TIMEOUT_SECONDS if the boot is just slow.",
                    getattr(spec.model_spec, "version", "?"),
                )
                break

        return CommandResult(
            command_name=self.name,
            return_code=1,
            error=(
                f"server bring-up failed after {attempts_made} attempt(s): {last_error}"
            ),
        )

    def _launch_once(self, spec: ServerLaunchSpec) -> Any:
        # Deferred: the seam's lazy default imports the adapter-side provider,
        # which imports this module for ServerMode.
        from .server_lifecycle import get_server_lifecycle

        return get_server_lifecycle().launch(
            spec.mode,
            spec.model_spec,
            spec.runtime_config,
            spec.setup_config,
            spec.json_fpath,
        )


class VenvCommand(Command):
    """Run an argv as a subprocess, optionally inside a declared workflow venv."""

    def __init__(
        self,
        venv_type: Any,
        argv: Sequence[str],
        *,
        model_spec: Any = None,
        env: Optional[Mapping[str, str]] = None,
        label: Optional[str] = None,
        dependency_venvs: Sequence[Any] = (),
    ) -> None:
        self.venv_type = venv_type
        self.argv = list(argv)
        self.model_spec = model_spec
        self.env = dict(env) if env is not None else None
        self.dependency_venvs = list(dependency_venvs)
        if label:
            self.name = label
        elif venv_type is None:
            self.name = "venv[current]"
        else:
            self.name = f"venv[{getattr(venv_type, 'name', venv_type)}]"

    def execute(self) -> CommandResult:
        import os
        import sys

        from .proc import run_command
        from .venv_provisioner import get_venv_provisioner

        if self.venv_type is None:
            python = sys.executable
        else:
            provisioner = get_venv_provisioner()

            # Provision the primary venv plus any dependency venvs the workflow
            # needs before running.
            for venv_type in [self.venv_type, *self.dependency_venvs]:
                if not provisioner.has_venv(venv_type):
                    return CommandResult(
                        command_name=self.name,
                        return_code=1,
                        error=f"no venv config for {venv_type!r}",
                    )
                if not provisioner.provision(venv_type, model_spec=self.model_spec):
                    return CommandResult(
                        command_name=self.name,
                        return_code=1,
                        error=(
                            f"failed to provision venv "
                            f"{getattr(venv_type, 'name', venv_type)}"
                        ),
                    )
            python = provisioner.venv_python(self.venv_type)

        cmd = [python, *[str(a) for a in self.argv]]
        env = {**os.environ, **self.env} if self.env else None
        try:
            return_code = run_command(cmd, logger=logger, env=env)
        except Exception as e:
            logger.exception("venv command failed: %s", e)
            return CommandResult(command_name=self.name, return_code=1, error=str(e))

        return CommandResult(
            command_name=self.name,
            return_code=return_code,
            error=None if return_code == 0 else f"exit code {return_code}",
        )


class WorkflowCommand(Command):
    name = "workflow"

    def __init__(
        self,
        ctx: MediaContext,
        *,
        workflow_name: str,
        orchestrator_metadata: OrchestratorMetadata,
        num_prompts: Optional[int] = None,
        continue_on_failure: bool = False,
    ) -> None:
        self.ctx = ctx
        self.workflow_name = workflow_name
        self.orchestrator_metadata = orchestrator_metadata
        self.num_prompts = num_prompts
        self.continue_on_failure = continue_on_failure

    def execute(self) -> CommandResult:
        from .blocks_sink import get_default_accumulator
        from .workflows import get_workflow_class

        self._apply_num_prompts_override()
        get_default_accumulator().clear()
        workflow_cls = get_workflow_class(self.workflow_name)
        workflow = workflow_cls(
            self.ctx,
            orchestrator_metadata=self.orchestrator_metadata,
        )
        result: WorkflowResult = workflow.run()
        return_code = result.return_code
        if return_code != 0 and self.continue_on_failure:
            logger.warning(
                "Workflow run failed (rc=%d, error=%s) but continuing because "
                "--repeat is active; this run is excluded from the summary.",
                return_code,
                result.error,
            )
            return_code = 0
        return CommandResult(
            command_name=self.name,
            return_code=return_code,
            error=result.error,
            payload=result,
        )

    def _apply_num_prompts_override(self) -> None:
        if self.num_prompts is None:
            return
        from test_module.benchmark_tests import image_benchmark_tests as _ibt

        _ibt.SDXL_BENCHMARK_NUM_PROMPTS = self.num_prompts
        _ibt.SDXL_SD35_BENCHMARK_NUM_PROMPTS = self.num_prompts
        logger.info(
            "Overriding image benchmark + spec_tests prompt count to %d",
            self.num_prompts,
        )


class SummaryCommand(Command):
    """Aggregate every per-run report under a container into one summary report."""

    name = "benchmark_summary"

    def __init__(self, *, container_dir: Path, summary_output_dir: Path) -> None:
        self.container_dir = container_dir
        self.summary_output_dir = summary_output_dir

    def execute(self) -> CommandResult:
        from .summary_report import summarize_container

        try:
            result = summarize_container(self.container_dir, self.summary_output_dir)
        except Exception as e:
            logger.exception("Benchmark summary failed: %s", e)
            return CommandResult(command_name=self.name, return_code=1, error=str(e))

        if result is None:
            logger.error(
                "No benchmark run reports found under %s — nothing to summarize.",
                self.container_dir,
            )
            return CommandResult(
                command_name=self.name, return_code=1, error="no_run_reports"
            )

        logger.info("Wrote benchmark summary: %s", result.markdown_path)
        return CommandResult(command_name=self.name, return_code=0, payload=result)


__all__ = [
    "Command",
    "CommandResult",
    "ServerCommand",
    "ServerLaunchSpec",
    "ServerMode",
    "SummaryCommand",
    "VenvCommand",
    "WorkflowCommand",
]
