# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

import sys
import runpy


def test_prompt_client_cli():
    # arguments
    args = {
        "num_prompts": "1",
        "batch_size": "32",
        "tokenizer_model": "meta-llama/Llama-3.1-70B-Instruct",
    }
    for key, value in args.items():
        if value is not None:
            sys.argv.extend(["--" + key, value])

    # runpy uses the same process and environment so the registered models are available
    runpy.run_module("utils.prompt_client_cli", run_name="cli_main")
