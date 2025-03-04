#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

original_dir=$PWD

if [[ -z "${HF_MODEL_REPO_ID}" ]]; then
    echo "⛔ Error: env var HF_MODEL_REPO_ID is not set. This must be the model HF repo e.g. 'meta-llama/Llama-3.3-70B-Instruct'"
    exit 1
fi

if [ -d "$HOME/llama-cookbook" ]; then
    echo "The directory $HOME/llama-recipes exists."
else
    echo "The directory ~/llama-recipes does not exist. Setting up ..."
    git clone https://github.com/meta-llama/llama-cookbook.git $HOME/llama-cookbook
    cd $HOME/llama-cookbook
    pip install -U pip setuptools
    pip install -e .
    pip install -U antlr4_python3_runtime==4.11
    pip install lm-eval[math,ifeval,sentencepiece,vllm]==0.4.3
    cd end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval
    python3 prepare_meta_eval.py --config_path ./eval_config.yaml
fi

# run evals
export OPENAI_API_KEY=$(python3 -c 'import os; import json; import jwt; json_payload = json.loads("{\"team_id\": \"tenstorrent\", \"token_id\": \"debug-test\"}"); encoded_jwt = jwt.encode(json_payload, os.environ["JWT_SECRET"], algorithm="HS256"); print(encoded_jwt)')
cd $HOME/llama-cookbook/end-to-end-use-cases/benchmarks/llm_eval_harness/meta_eval

# IFEval
lm_eval \
--model local-completions \
--model_args model=${HF_MODEL_REPO_ID},base_url=http://127.0.0.1:7000/v1,tokenizer_backend=huggingface \
--gen_kwargs stream=False \
--tasks meta_ifeval \
--include_path ./work_dir \
--output_path eval_output \
--seed 42  \
--log_samples

# GPQA
lm_eval \
--model local-completions \
--model_args model=${HF_MODEL_REPO_ID},base_url=http://127.0.0.1:7000/v1,tokenizer_backend=huggingface \
--gen_kwargs stream=False \
--tasks meta_gpqa \
--include_path ./work_dir \
--output_path eval_output \
--seed 42  \
--log_samples

# GPQA CoT
lm_eval \
--model local-completions \
--model_args model=${HF_MODEL_REPO_ID},base_url=http://127.0.0.1:7000/v1,tokenizer_backend=huggingface \
--gen_kwargs stream=False \
--tasks meta_gpqa_cot \
--include_path ./work_dir \
--output_path eval_output \
--seed 42  \
--log_samples

# MMLU pro
lm_eval \
--model local-completions \
--model_args model=${HF_MODEL_REPO_ID},base_url=http://127.0.0.1:7000/v1,tokenizer_backend=huggingface \
--gen_kwargs stream=False \
--tasks meta_mmlu_pro_instruct \
--include_path ./work_dir \
--output_path eval_output \
--seed 42  \
--log_samples

cd $original_dir
