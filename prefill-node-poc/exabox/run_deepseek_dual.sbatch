#!/bin/bash
#SBATCH --job-name=deepseek_dual
#SBATCH --partition=wh_pod_8x16_2
#SBATCH --nodes=2
#SBATCH --nodelist=wh-glx-a05u02,wh-glx-a05u08
#SBATCH --output=deepseek_%j.out
#SBATCH --error=deepseek_%j.err
#SBATCH --exclusive

# =============================================================================
# DeepSeek-V3/R1 Demo on Dual Galaxy (Exabox Slurm)
# =============================================================================
# Usage:
#   sbatch slurm/run_deepseek_dual.sbatch
#
# Monitor:
#   tail -f deepseek_<jobid>.out deepseek_<jobid>.err
# =============================================================================

set -e

# === Configuration (Update these paths for your setup) ===
export TT_METAL_HOME="/data/dmadic/tt-metal"
export PYTHONPATH="${TT_METAL_HOME}"

# Model paths - adjust to your model location on exabox
export DEEPSEEK_V3_HF_MODEL="/data/models/deepseek-ai/DeepSeek-R1-0528"
export DEEPSEEK_V3_CACHE="/data/dmadic/deepseek_cache"

# Mesh configuration for Dual Galaxy (8x8 = 64 devices across 2 hosts)
export MESH_DEVICE="DUAL"

# POC directory
POC_DIR="/data/dmadic/tt-inference-server/prefill-node-poc"

# === Slurm host discovery ===
HOSTFILE="${SLURM_JOB_ID}_hosts.txt"
scontrol show hostnames $SLURM_JOB_NODELIST > ${HOSTFILE}
HOSTS=$(cat ${HOSTFILE} | paste -sd,)

echo "=============================================="
echo "DeepSeek Dual Galaxy Job: ${SLURM_JOB_ID}"
echo "Running on hosts: ${HOSTS}"
echo "TT_METAL_HOME: ${TT_METAL_HOME}"
echo "MESH_DEVICE: ${MESH_DEVICE}"
echo "=============================================="

# Rank binding file (uses absolute path for MGD)
RANK_BINDING="${POC_DIR}/slurm/exabox_dual_galaxy_rank_bindings.yaml"

# MPI arguments for multi-host execution
MPI_ARGS="--hostfile ${HOSTFILE} --map-by ppr:1:node --mca btl self,tcp --mca btl_tcp_if_exclude docker0,lo --bind-to none --tag-output"

# === Activate environment ===
cd ${TT_METAL_HOME}
source python_env/bin/activate

# === Reset devices on all nodes (optional but recommended) ===
echo ""
echo "=== Resetting devices on all nodes ==="
mpirun -np 2 --hostfile ${HOSTFILE} --map-by ppr:1:node \
    bash -c "tt-smi -r 2>/dev/null || echo 'Reset skipped on \$(hostname)'"
sleep 5

# === Run DeepSeek Demo ===
echo ""
echo "=== Running DeepSeek Demo ==="
echo "Rank binding: ${RANK_BINDING}"
echo ""

tt-run --verbose \
    --rank-binding ${RANK_BINDING} \
    --mpi-args "${MPI_ARGS}" \
    python models/demos/deepseek_v3/demo/demo.py \
        --model-path ${DEEPSEEK_V3_HF_MODEL} \
        --cache-dir ${DEEPSEEK_V3_CACHE} \
        --max-new-tokens 32 \
        --early_print_first_user \
        "Write a haiku about tensors flowing through silicon"

# === Cleanup ===
rm -f ${HOSTFILE}

echo ""
echo "=== Job completed ==="
