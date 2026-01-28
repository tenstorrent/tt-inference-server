# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import logging
import shutil
import sys

from workflows.utils import get_repo_root_path, run_command

logger = logging.getLogger("run_log")

UV_VENV_PATH = get_repo_root_path() / ".workflow_venvs" / ".venv_bootstrap_uv"
UV_EXEC = UV_VENV_PATH / "bin" / "uv"


def bootstrap_uv():
    """Bootstrap uv package manager by creating a venv and installing uv via pip.

    This function creates a dedicated virtual environment for uv and installs
    the uv package manager. The UV_EXEC constant can be imported and used
    after calling this function.
    """
    # Step 1: Check Python version
    python_version = sys.version_info
    if python_version < (3, 6):
        raise ValueError("Python 3.6 or higher is required.")

    logger.info(
        "Python version: %d.%d.%d",
        python_version.major,
        python_version.minor,
        python_version.micro,
    )

    # Step 2: Create a virtual environment
    pip_exec = UV_VENV_PATH / "bin" / "pip"
    venv_python = UV_VENV_PATH / "bin" / "python"

    # Check if venv needs to be created or recreated (e.g., if pip is missing)
    needs_venv_creation = not UV_VENV_PATH.exists() or not pip_exec.exists()

    if not needs_venv_creation:
        logger.info(f"uv bootstrap venv already exists at: {UV_VENV_PATH}")
        return

    logger.info(f"Creating uv bootstrap venv at: {UV_VENV_PATH}")
    # Clear existing venv if it exists but is broken (missing pip)
    if UV_VENV_PATH.exists():
        shutil.rmtree(UV_VENV_PATH)

    # Create venv - some systems (PEP 668 externally-managed) may not include pip
    run_command(
        f"{sys.executable} -m venv {UV_VENV_PATH}",
        logger=logger,
    )

    # Ensure pip is installed using ensurepip (works even on externally-managed Python)
    if not pip_exec.exists():
        logger.info("Installing pip using ensurepip...")
        run_command(
            f"{venv_python} -m ensurepip --upgrade",
            logger=logger,
        )

    # Step 3: Install 'uv' using pip
    # Note: Activating the virtual environment in a script doesn't affect the current shell,
    # so we directly use the pip executable from the venv.
    logger.info("Installing 'uv' using pip...")
    run_command(f"{pip_exec} install uv", logger=logger)

    logger.info("uv bootstrap installation complete.")
    # check version
    run_command(f"{str(UV_EXEC)} --version", logger=logger)
