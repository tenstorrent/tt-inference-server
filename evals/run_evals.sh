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
    pip install git+https://github.com/tstescoTT/lm-evaluation-harness.git#egg=lm-eval[ifeval]
fi

if [ -d "$HOME/llama-recipes" ]; then
    echo "The directory $HOME/llama-recipes exists."
else
    echo "The directory ~/llama-recipes does not exist."
    git clone https://github.com/tstescoTT/llama-recipes.git $HOME/llama-recipes
    cd $HOME/llama-recipes/tools/benchmarks/llm_eval_harness/meta_eval
    python prepare_meta_eval.py --config_path ./eval_config.yaml
    mkdir -p $HOME/lm-evaluation-harness
    cp -rf work_dir/ $HOME/lm-evaluation-harness/
fi

# trace capture so that evals do not trigger 1st run trace capture unexpectedly
cd $HOME/app
python utils/capture_traces.py

# run evals
export OPENAI_API_KEY=$(python -c 'import os; import json; import jwt; json_payload = json.loads("{\"team_id\": \"tenstorrent\", \"token_id\": \"debug-test\"}"); encoded_jwt = jwt.encode(json_payload, os.environ["JWT_SECRET"], algorithm="HS256"); print(encoded_jwt)')
cd $HOME/lm-evaluation-harness/

# GPQA
lm_eval \
--model local-completions \
--model_args model=${HF_MODEL_REPO_ID},base_url=http://127.0.0.1:7000/v1/completions,num_concurrent=32,max_retries=4,tokenized_requests=False,add_bos_token=True,timeout=None \
--gen_kwargs model=${HF_MODEL_REPO_ID},stop="<|eot_id|>",stream=False \
--tasks meta_gpqa \
--batch_size auto \
--output_path ${CACHE_ROOT}/eval_output \
--include_path ./work_dir \
--seed 42  \
--log_samples

# IFEval
lm_eval \
--model local-completions \
--model_args model=${HF_MODEL_REPO_ID},base_url=http://127.0.0.1:7000/v1/completions,num_concurrent=32,max_retries=4,tokenized_requests=False,add_bos_token=True,timeout=None \
--gen_kwargs model=${HF_MODEL_REPO_ID},stop="<|eot_id|>",stream=False \
--tasks meta_ifeval \
--batch_size auto \
--output_path ${CACHE_ROOT}/eval_output \
--include_path ./work_dir \
--seed 42  \
--log_samples

cd $original_dir
