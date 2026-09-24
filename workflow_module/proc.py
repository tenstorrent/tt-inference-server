# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

"""Subprocess helpers for the engine core.

Moved here from ``workflows/utils.py`` so engine packages never import the
Tenstorrent adapter for a generic subprocess wrapper. ``workflows.utils``
re-exports these for pre-extraction callers.
"""

from __future__ import annotations

import logging
import math
import os
import shlex
import signal
import subprocess
import threading
import time

logger = logging.getLogger(__name__)


def stream_subprocess_output(pipe, logger, level):
    with pipe:
        for line in iter(pipe.readline, ""):
            logger.log(level, line.strip(), extra={"raw": True})


def run_command(
    command,
    logger,
    log_file_path=None,
    shell=False,
    copy_env=True,
    env=None,
    check=False,
    timeout_seconds=None,
):
    """
    This function is a wrapper around subprocess.Popen and subprocess.run.
    It is used to run a command and capture the stdout and stderr in the caller's logger.

    Args:
        command: Command to run. Can be a string or a list of strings.
        logger: Logger to use for logging. Must be passed because the common use case is to capture the command's stdout and stderr in the caller's logger.
        log_file_path: Path to log file. If None, stdout and stderr will be logged to the logger.
        shell: Whether to use shell to run the command.
        copy_env: Whether to copy the environment variables.
        env: Environment variables to use.
        check: Whether to check the return code. Set to True for commands that must succeed.
    Returns:
        Return code of the command.
    Raises:
        RuntimeError: If the command fails and check is True.
        NotImplementedError: If copy_env is True and not implemented.
        AssertionError: If command is not a list of strings.
        ValueError: If command is None.
        PermissionError: If the directory is not writable.
        IOError: If the directory is not readable.
    """
    if not copy_env:
        raise NotImplementedError("TODO")

    if not env:
        env = os.environ.copy()
    # TODO: force usage to always use argument list
    # use shlex to log full command before running

    if command is None:
        logger.error("No command provided to run_command.")
    elif isinstance(command, str):
        command = shlex.split(command)

    assert isinstance(command, list), "Command must be a list of cmd arguments."

    if timeout_seconds is not None:
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        if os.name != "posix":
            raise NotImplementedError("Bounded process-tree execution requires POSIX")
        return_code = _run_bounded_command(
            command, logger, env, shell, log_file_path, timeout_seconds
        )
        if check and return_code:
            raise RuntimeError(f"Bounded command failed with return code {return_code}")
        return return_code

    logger.info(f"Running command: {shlex.join(command)}")

    if not log_file_path:
        subproc_type = "subprocess.Popen"
        # capture all output to stdout and stderr in current process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
            env=env,
        )

        stdout_thread = threading.Thread(
            target=stream_subprocess_output,
            args=(process.stdout, logger, logging.DEBUG),
        )
        stderr_thread = threading.Thread(
            target=stream_subprocess_output,
            args=(process.stderr, logger, logging.INFO),
        )

        stdout_thread.start()
        stderr_thread.start()

        stdout_thread.join()
        stderr_thread.join()

        process.wait()
        return_code = process.returncode
    else:
        subproc_type = "subprocess.run"
        logger.info(f"Logging output to: {log_file_path} ...")
        with open(log_file_path, "a", buffering=1) as log_file:
            result = subprocess.run(
                command,
                shell=shell,
                stdout=log_file,
                stderr=log_file,
                check=check,
                text=True,
                env=env,
            )
            return_code = result.returncode

    if return_code != 0:
        error_message = (
            f"⛔ {subproc_type} command failed with return code: {return_code}\n"
            f"command: {shlex.join(command)}\n\n"
            "See error messages in logs above this RuntimeError for details on actual cause of failure.\n"
        )
        if check:
            raise RuntimeError(error_message)
        else:
            logger.error(
                error_message
                + "\nThis command is optional or can be recovered from failure (check=False set). Continuing ...\n"
            )
    return return_code


def _run_bounded_command(command, logger, env, shell, log_file_path, timeout_seconds):
    """Bound the complete process group, including children holding log pipes.

    Preserve emitted logs/files. Exit 124 denotes an incomplete timed-out task,
    not an evaluation score. Existing callers remain unbounded unless opted in.
    """
    logger.info("Running command: %s", shlex.join(command))
    if log_file_path:
        logger.info("Logging output to: %s ...", log_file_path)
    log_file = open(log_file_path, "a", buffering=1) if log_file_path else None
    process = None
    readers = []
    deadline = time.monotonic() + timeout_seconds
    try:
        process = subprocess.Popen(
            command,
            shell=shell,
            env=env,
            start_new_session=True,
            stdout=log_file or subprocess.PIPE,
            stderr=log_file or subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if not log_file:
            for pipe, level in (
                (process.stdout, logging.DEBUG),
                (process.stderr, logging.INFO),
            ):
                reader = threading.Thread(
                    target=stream_subprocess_output,
                    args=(pipe, logger, level),
                    daemon=True,
                )
                reader.start()
                readers.append(reader)
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
            for reader in readers:
                reader.join(timeout=max(0, deadline - time.monotonic()))
            if any(reader.is_alive() for reader in readers):
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            return process.returncode
        except subprocess.TimeoutExpired:
            logger.error(
                "Evaluation execution deadline exceeded (%ss); incomplete, rc=124",
                timeout_seconds,
            )
            return 124
    finally:
        # Kill the owned process group even if its leader exited before children.
        # No host-wide process matching, and no inference server termination.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        for reader in readers:
            reader.join(timeout=2)
        if log_file:
            log_file.close()
