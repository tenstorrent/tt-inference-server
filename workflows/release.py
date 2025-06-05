# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import time
import shlex
import subprocess

from workflows.model_config import MODEL_CONFIGS
from evals.eval_config import EVAL_CONFIGS

from workflows.utils import get_repo_root_path


def main():
    run_script_path = get_repo_root_path() / "run.py"
    available_evals = set(EVAL_CONFIGS.keys())
    for model_id, model_config in MODEL_CONFIGS.items():
        if model_config.model_name not in available_evals:
            print(f"Model {model_config.model_name} does not have evals")
            continue
        device = model_config.device_type.name.lower()
        print(f"running Model: {model_config.model_name} on device: {device} ...")
        # fmt: off
        args = [
            "python3", str(run_script_path),
            "--model", model_config.model_name,
            "--device", device,
            "--workflow", "release",
            "--dev-mode",
            "--docker-server",
        ]
        # fmt: on
        print(f"running: {shlex.join(args)}")
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
