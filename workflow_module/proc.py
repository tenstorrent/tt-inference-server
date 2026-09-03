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
import os
import shlex
import subprocess
import threading

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
