#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# GPQA
lm_eval \
--model local-completions \
--model_args model=meta-llama/Llama-3.1-70B-Instruct,base_url=http://127.0.0.1:7000/v1/completions,num_concurrent=16,max_retries=4,tokenized_requests=False,add_bos_token=True \
--gen_kwargs model=meta-llama/Llama-3.1-70B-Instruct,stop="<|eot_id|>",stream=False \
--tasks meta_gpqa \
--batch_size auto \
--output_path /home/user/cache_root/eval_output \
--include_path ./work_dir \
--seed 42  \
--log_samples

# IFEval
lm_eval \
--model local-completions \
--model_args model=meta-llama/Llama-3.1-70B-Instruct,base_url=http://127.0.0.1:7000/v1/completions,num_concurrent=32,max_retries=4,tokenized_requests=False,add_bos_token=True \
--gen_kwargs model=meta-llama/Llama-3.1-70B-Instruct,stop="<|eot_id|>",stream=False \
--tasks meta_ifeval \
--batch_size auto \
--output_path /home/user/cache_root/eval_output \
--include_path ./work_dir \
--seed 42  \
--log_samples
