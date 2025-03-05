#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

original_dir=$PWD

if [[ -z "${HF_MODEL_REPO_ID}" ]]; then
    echo "⛔ Error: env var HF_MODEL_REPO_ID is not set. This must be the model HF repo e.g. 'meta-llama/Llama-3.3-70B-Instruct'"
    exit 1
fi

if [ ! -d "$HOME/lm-evaluation-harness" ]; then
    cd $HOME
    git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
    cd lm-evaluation-harness
    git checkout 6d2abda4fd171e68a8789330c4149e37c1ca0bda
    pip install -e .
fi

# run evals
export OPENAI_API_KEY=$(python3 -c 'import os; import json; import jwt; json_payload = json.loads("{\"team_id\": \"tenstorrent\", \"token_id\": \"debug-test\"}"); encoded_jwt = jwt.encode(json_payload, os.environ["JWT_SECRET"], algorithm="HS256"); print(encoded_jwt)')

cd $original_dir
