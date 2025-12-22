# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import importlib
import os
import subprocess
from .forge_runner import ForgeRunner

"""
Forge model runners for different CNN architectures.
Loaders are dynamically imported from the corresponding loader modules.
"""


class ForgeMobilenetv2Runner(ForgeRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.loader = load_dynamic("mobilenetv2")


class ForgeResnetRunner(ForgeRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.loader = load_dynamic("resnet")


class ForgeVovnetRunner(ForgeRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.loader = load_dynamic("vovnet")


class ForgeEfficientnetRunner(ForgeRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.loader = load_dynamic("efficientnet")


class ForgeSegformerRunner(ForgeRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.loader = load_dynamic("segformer")


class ForgeUnetRunner(ForgeRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.loader = load_dynamic("unet")


class ForgeVitRunner(ForgeRunner):
    def __init__(self, device_id: str):
        super().__init__(device_id)
        self.loader = load_dynamic("vit")


def ensure_model_loaders():
    """Ensure model_loaders directory exists by running fetch_models.sh if needed."""
    current_dir = os.path.dirname(__file__)
    model_loaders_path = os.path.join(current_dir, "model_loaders")
    if not os.path.exists(model_loaders_path):
        fetch_script_path = os.path.join(current_dir, "fetch_models.sh")
        subprocess.run(
            [fetch_script_path],
            cwd=current_dir,
            check=True,
            capture_output=True,
            text=True,
        )


def load_dynamic(model_name: str):
    # Ensure model loaders are available
    ensure_model_loaders()

    try:
        # Import from the forge_runners package, not from this module
        module_path = (
            f"tt_model_runners.forge_runners.model_loaders.{model_name}.pytorch.loader"
        )
        module = importlib.import_module(module_path)
        return module.ModelLoader()
    except ImportError as e:
        raise ImportError(f"Failed to load {module_path} module: {e}")
    except AttributeError as e:
        raise AttributeError(
            f"ModelLoader class not found in {module_path} loader module: {e}"
        )
