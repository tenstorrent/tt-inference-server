#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
TT-NN Visualizer UI process manager for tt-inference-server.

Manages starting and stopping the ttnn-visualizer web UI as a background process.
"""

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def find_ttnn_visualizer_repo() -> Optional[Path]:
    """
    Find ttnn-visualizer repository.
    
    Searches in common locations:
    1. Environment variable TTNN_VISUALIZER_PATH
    2. Sibling directory to tt-inference-server
    3. Parent directory
    
    Returns:
        Path to ttnn-visualizer repo or None if not found
    """
    # Check environment variable
    env_path = os.getenv("TTNN_VISUALIZER_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists() and (path / "package.json").exists():
            return path
    
    # Check sibling directory
    repo_root = Path(__file__).resolve().parent.parent
    sibling_path = repo_root.parent / "ttnn-visualizer"
    if sibling_path.exists() and (sibling_path / "package.json").exists():
        return sibling_path
    
    # Check parent directory
    parent_path = repo_root.parent / "ttnn-visualizer"
    if parent_path.exists() and (parent_path / "package.json").exists():
        return parent_path
    
    return None


def check_node_installed() -> bool:
    """Check if Node.js and npm are installed."""
    try:
        subprocess.run(
            ["node", "--version"],
            capture_output=True,
            check=True
        )
        subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_npm_deps_installed(visualizer_path: Path) -> bool:
    """Check if npm dependencies are installed."""
    node_modules = visualizer_path / "node_modules"
    return node_modules.exists() and node_modules.is_dir()


def install_npm_deps(visualizer_path: Path) -> bool:
    """Install npm dependencies for visualizer."""
    logger.info("Installing npm dependencies for ttnn-visualizer...")
    try:
        subprocess.run(
            ["npm", "install"],
            cwd=visualizer_path,
            capture_output=True,
            check=True,
            text=True
        )
        logger.info("✅ npm dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install npm dependencies: {e.stderr}")
        return False


def start_visualizer_ui(reports_dir: Path, port: int = 5173) -> Optional[subprocess.Popen]:
    """
    Start ttnn-visualizer web UI as a background process.
    
    Args:
        reports_dir: Path to directory containing reports
        port: Port number for web UI (default: 5173, Vite's default)
        
    Returns:
        Popen process object or None if failed to start
    """
    # Check if Node.js is installed
    if not check_node_installed():
        logger.warning("⚠️  Node.js/npm not found. Cannot start ttnn-visualizer UI.")
        logger.warning("    Install Node.js: https://nodejs.org/")
        logger.warning("    Or view reports manually: cd ttnn-visualizer && npm run dev")
        return None
    
    # Find visualizer repo
    visualizer_path = find_ttnn_visualizer_repo()
    if not visualizer_path:
        logger.warning("⚠️  ttnn-visualizer repository not found.")
        logger.warning("    Set TTNN_VISUALIZER_PATH environment variable")
        logger.warning("    Or place ttnn-visualizer as sibling to tt-inference-server")
        logger.warning(f"    Reports saved to: {reports_dir}")
        return None
    
    logger.info(f"Found ttnn-visualizer at: {visualizer_path}")
    
    # Check if port is available
    if is_visualizer_ui_running(port):
        logger.warning(f"⚠️  Port {port} is already in use.")
        logger.warning(f"    A visualizer UI may already be running at http://localhost:{port}")
        logger.warning(f"    Or use a different port: --ttnn-visualizer-port <port>")
        logger.info(f"    Reports saved to: {reports_dir}")
        return None
    
    # Check if dependencies are installed
    if not check_npm_deps_installed(visualizer_path):
        logger.info("npm dependencies not found, installing...")
        if not install_npm_deps(visualizer_path):
            logger.error("Failed to install npm dependencies")
            return None
    
    # Start the dev server in background
    logger.info(f"Starting ttnn-visualizer web UI on port {port}...")
    try:
        # Start npm run dev with custom port as background process
        # Vite accepts --port flag to override default 5173
        process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(port)],
            cwd=visualizer_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Detach from parent process group to allow it to run independently
            preexec_fn=os.setpgrp if os.name != 'nt' else None
        )
        
        # Give it a moment to start
        time.sleep(2)
        
        # Check if process started successfully
        if process.poll() is not None:
            # Process died immediately
            stdout, stderr = process.communicate()
            logger.error(f"Failed to start visualizer UI: {stderr}")
            return None
        
        logger.info("✅ TT-NN Visualizer UI started successfully")
        logger.info(f"   Web UI: http://localhost:{port}")
        logger.info("   The UI will remain open in the background")
        logger.info("   Load reports from the UI by navigating to the reports directory")
        
        return process
        
    except Exception as e:
        logger.error(f"Failed to start visualizer UI: {e}")
        return None


def stop_visualizer_ui(process: subprocess.Popen) -> None:
    """
    Stop ttnn-visualizer web UI process.
    
    Args:
        process: Popen process object returned from start_visualizer_ui
    """
    if process is None:
        return
    
    try:
        logger.info("Stopping ttnn-visualizer web UI...")
        process.terminate()
        
        # Wait up to 5 seconds for graceful shutdown
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if it doesn't stop
            process.kill()
            process.wait()
        
        logger.info("✅ TT-NN Visualizer UI stopped")
        
    except Exception as e:
        logger.warning(f"Error stopping visualizer UI: {e}")


def is_visualizer_ui_running(port: int = 5173) -> bool:
    """
    Check if visualizer UI is already running on the specified port.
    
    Args:
        port: Port number to check (default: 5173)
        
    Returns:
        True if port is in use, False otherwise
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("localhost", port))
        sock.close()
        return False
    except OSError:
        # Port is in use
        return True

