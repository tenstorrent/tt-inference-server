#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

original_dir=$PWD

if [[ -z "${HF_MODEL_REPO_ID}" ]]; then
    echo "⛔ Error: env var HF_MODEL_REPO_ID is not set. This must be the model HF repo e.g. 'meta-llama/Llama-3.3-70B-Instruct'"
    exit 1
fi

# set up lm_eval and evals datasets
cd $HOME
if python -c "import lm_eval" 2>/dev/null; then
    echo "lm_eval is installed."
else
    echo "Installing lm_eval ..."
    pip install git+https://github.com/tstescoTT/lm-evaluation-harness.git@tstesco/add-local-multimodal#egg=lm-eval
fi

# trace capture so that evals do not trigger 1st run trace capture unexpectedly
# cd $HOME/app
# TODO: add support for vision model in capture_traces.py
# python utils/capture_traces.py

# run evals
export OPENAI_API_KEY=$(python -c 'import os; import json; import jwt; json_payload = json.loads("{\"team_id\": \"tenstorrent\", \"token_id\": \"debug-test\"}"); encoded_jwt = jwt.encode(json_payload, os.environ["JWT_SECRET"], algorithm="HS256"); print(encoded_jwt)')
# cd $HOME/lm-evaluation-harness/

# MMMU
lm_eval \
--model local-mm-chat-completions \
--model_args pretrained=${HF_MODEL_REPO_ID},base_url=http://127.0.0.1:7000/v1/chat/completions,num_concurrent=16,max_retries=4,tokenized_requests=False,add_bos_token=True,timeout=9999,eos_string="<|end_of_text|>" \
--gen_kwargs model=${HF_MODEL_REPO_ID},stop="<|eot_id|>",stream=False \
--tasks mmmu_val \
--batch_size auto \
--output_path /home/user/cache_root/eval_output \
--seed 42  \
--log_samples

cd $original_dir