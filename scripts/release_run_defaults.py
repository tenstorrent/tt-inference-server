# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import time
import shlex
import subprocess
import sys
from pathlib import Path

# Add the script's directory to the Python path
# this for 0 setup python setup script
project_root = Path(__file__).resolve().parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.utils import get_repo_root_path
from workflows.model_config import config_templates, model_weights_to_model_name
from evals.eval_config import EVAL_CONFIGS


def main():
    run_script_path = get_repo_root_path() / "run.py"
    available_evals = set(EVAL_CONFIGS.keys())
    for config_template in config_templates:
        model_name = model_weights_to_model_name(config_template.weights[0])
        if model_name not in available_evals:
            # check for instruct model with evals
            model_name = f"{model_name}-Instruct"
            if model_name not in available_evals:
                print(f"Model {model_name} does not have evals")
                continue
            else:
                print(f"Using instruct version for evals: {model_name}")
        # only run default model-hardware-impl combinations
        default_hardware = {
            device
            for device, dev_spec in config_template.device_model_spec_map.items()
            if dev_spec.default_impl
        }
        for device in default_hardware:
            device_str = device.name.lower()
            # fmt: off
            args = [
                "python3", str(run_script_path),
                "--model", model_name,
                "--device", device_str,
                "--workflow", "release",
                "--dev-mode",
                "--docker-server",
            ]
            # fmt: on
            print(f"running command: {shlex.join(args)}")
            subprocess.run(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=False,
            )
            time.sleep(2)


if __name__ == "__main__":
    main()
