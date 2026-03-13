#!/bin/bash
#SBATCH --job-name=prefill_real
#SBATCH --partition=wh_pod_8x16_2
#SBATCH --nodes=2
#SBATCH --nodelist=wh-glx-a05u02,wh-glx-a05u08
#SBATCH --output=prefill_real_%j.out
#SBATCH --error=prefill_real_%j.err
#SBATCH --exclusive

# =============================================================================
# DeepSeek-V3/R1 Prefill Demo - vLLM-like Engine (Dual Galaxy / Exabox)
# =============================================================================
# This script runs prefill on the real DeepSeek model using the vLLM-like
# engine infrastructure (scheduler, block manager, real model runner).
# Prints the first token (greedy sampled) for each prompt.
#
# Usage:
#   sbatch exabox/run_prefill_real.sbatch
#
# Monitor:
#   tail -f prefill_real_<jobid>.out prefill_real_<jobid>.err
# =============================================================================

set -e

# === Configuration (Update these paths for your setup) ===
export TT_METAL_HOME="/data/dmadic/tt-metal"

# POC directory
POC_DIR="/data/dmadic/tt-inference-server/prefill-node-poc"

# PYTHONPATH: include both tt-metal and POC directory for imports
export PYTHONPATH="${TT_METAL_HOME}:${POC_DIR}:${PYTHONPATH}"

# Model paths - adjust to your model location on exabox
export DEEPSEEK_V3_HF_MODEL="/mnt/models/deepseek-ai/DeepSeek-R1-0528"
export DEEPSEEK_V3_CACHE="/data/dmadic/deepseek-ai/DeepSeek-R1-0528-Cache"

# Mesh configuration for Dual Galaxy (8x8 = 64 devices across 2 hosts)
export MESH_DEVICE="DUAL"

# === Slurm host discovery ===
HOSTFILE="/tmp/${SLURM_JOB_ID}_hosts.txt"
scontrol show hostnames $SLURM_JOB_NODELIST > ${HOSTFILE}
HOSTS=$(cat ${HOSTFILE} | paste -sd,)

echo "=============================================="
echo "DeepSeek Prefill Demo Job: ${SLURM_JOB_ID}"
echo "Running on hosts: ${HOSTS}"
echo "TT_METAL_HOME: ${TT_METAL_HOME}"
echo "MESH_DEVICE: ${MESH_DEVICE}"
echo "=============================================="

# Rank binding file (uses absolute path for MGD)
RANK_BINDING="${POC_DIR}/exabox/exabox_dual_galaxy_rank_bindings.yaml"

# MPI arguments for multi-host execution
MPI_ARGS="--hostfile ${HOSTFILE} --map-by ppr:1:node --mca btl self,tcp --mca btl_tcp_if_exclude docker0,lo --bind-to none --tag-output"

# === Activate environment ===
cd ${TT_METAL_HOME}
source python_env/bin/activate

# === Run Prefill Demo ===
echo ""
echo "=== Running DeepSeek Prefill Demo (vLLM-like Engine) ==="
echo "Rank binding: ${RANK_BINDING}"
echo ""

# Run with default prompts (can customize by adding prompts as arguments)
tt-run --verbose \
    --rank-binding ${RANK_BINDING} \
    --mpi-args "${MPI_ARGS}" \
    python ${POC_DIR}/main_prefill_real.py \
        --model-path ${DEEPSEEK_V3_HF_MODEL} \
        --cache-dir ${DEEPSEEK_V3_CACHE} \
        "What is the capital of France?"

# === Cleanup ===
rm -f ${HOSTFILE}

echo ""
echo "=== Job completed ==="
