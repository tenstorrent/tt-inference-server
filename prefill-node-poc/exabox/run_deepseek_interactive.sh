#!/bin/bash
# =============================================================================
# DeepSeek-V3/R1 Interactive Launch Script (Exabox Slurm)
# =============================================================================
# Usage:
#   1. Get a slurm allocation:
#      srun -p wh_pod_8x16_2 -N 2 --nodelist=wh-glx-a05u02,wh-glx-a05u08 --pty /bin/bash
#
#   2. Run this script:
#      ./exabox/run_deepseek_interactive.sh "Your prompt here"
#
#   3. Or with a prompts file:
#      ./exabox/run_deepseek_interactive.sh --prompts-file ${TT_METAL_HOME}/models/demos/deepseek_v3/demo/test_prompts.json
# =============================================================================

set -e

# === Configuration (Update these paths for your setup) ===
export TT_METAL_HOME="/data/dmadic/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}"

# Model paths
export DEEPSEEK_V3_HF_MODEL="/mnt/models/deepseek-ai/DeepSeek-R1-0528"
export DEEPSEEK_V3_CACHE="/data/dmadic/deepseek-ai/DeepSeek-R1-0528-Cache"

# Mesh configuration
export MESH_DEVICE="DUAL"

# POC directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POC_DIR="$(dirname "${SCRIPT_DIR}")"

# === Validate slurm environment ===
if [ -z "$SLURM_JOB_NODELIST" ]; then
    echo "ERROR: Not running inside a slurm allocation."
    echo "First get an allocation:"
    echo "  srun -p wh_pod_8x16_2 -N 2 --nodelist=wh-glx-a05u02,wh-glx-a05u08 --pty /bin/bash"
    exit 1
fi

# === Get hosts from slurm ===
HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)
NUM_HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)

echo "=============================================="
echo "DeepSeek Interactive Session"
echo "Running on ${NUM_HOSTS} hosts: ${HOSTS}"
echo "MESH_DEVICE: ${MESH_DEVICE}"
echo "=============================================="

if [ "$NUM_HOSTS" -ne 2 ]; then
    echo "WARNING: Expected 2 hosts for Dual Galaxy, got ${NUM_HOSTS}"
fi

# === Setup ===
cd ${TT_METAL_HOME}
source python_env/bin/activate

RANK_BINDING="${POC_DIR}/exabox/exabox_dual_galaxy_rank_bindings.yaml"
MPI_ARGS="--host ${HOSTS} --map-by ppr:1:node --mca btl self,tcp --mca btl_tcp_if_exclude docker0,lo --bind-to none --tag-output"

# === Parse arguments ===
DEMO_ARGS=""
if [ $# -eq 0 ]; then
    # Default prompt if none provided
    DEMO_ARGS="\"Hello, I am DeepSeek running on Tenstorrent hardware!\""
else
    # Pass all arguments to the demo
    DEMO_ARGS="$@"
fi

# === Run ===
echo ""
echo "Running with args: ${DEMO_ARGS}"
echo ""

tt-run --verbose \
    --rank-binding ${RANK_BINDING} \
    --mpi-args "${MPI_ARGS}" \
    python models/demos/deepseek_v3/demo/demo.py \
        --model-path ${DEEPSEEK_V3_HF_MODEL} \
        --cache-dir ${DEEPSEEK_V3_CACHE} \
        --max-new-tokens 32 \
        --early_print_first_user \
        ${DEMO_ARGS}
