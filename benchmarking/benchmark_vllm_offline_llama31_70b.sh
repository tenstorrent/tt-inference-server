# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# for Time To First Token (TTFT) and throughput (TPS, TPOT, ITL)
python ~/tests/benchmark_vllm_offline_inference.py  --max_seqs_in_batch 32 --input_seq_len 128 --output_seq_len 128 --greedy_sampling
python ~/tests/benchmark_vllm_offline_inference.py  --max_seqs_in_batch 32 --input_seq_len 512 --output_seq_len 512 --greedy_sampling
python ~/tests/benchmark_vllm_offline_inference.py  --max_seqs_in_batch 32 --input_seq_len 1024 --output_seq_len 1024 --greedy_sampling
python ~/tests/benchmark_vllm_offline_inference.py  --max_seqs_in_batch 32 --input_seq_len 2048 --output_seq_len 2048 --greedy_sampling

