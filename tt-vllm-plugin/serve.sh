# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

VLLM_USE_V1=1 HF_MODEL='meta-llama/Llama-3.1-8B-Instruct' \
vllm serve 'meta-llama/Llama-3.1-8B-Instruct' \
    --max-model-len 65536 \
    --max-num-seqs 32 \
    --block-size 64 \
    --max-num-batched-tokens 65536