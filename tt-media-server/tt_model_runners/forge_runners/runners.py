# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import importlib
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


def load_dynamic(model_name: str):
    try:
        # Import from the forge_runners package, not from this module
        module_path = f"tt_model_runners.forge_runners.loaders.{model_name}.pytorch.loader"
        module = importlib.import_module(module_path)
        return module.ModelLoader()
    except ImportError as e:
        raise ImportError(f"Failed to load {module_path} module: {e}")
    except AttributeError as e:
        raise AttributeError(f"ModelLoader class not found in {module_path} loader module: {e}")