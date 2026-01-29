#!/bin/bash
# =============================================================================
# Reset Tenstorrent Devices on Slurm Nodes
# =============================================================================
# Usage:
#   1. Get a slurm allocation:
#      srun -p wh_pod_8x16_2 -N 2 --nodelist=wh-glx-a05u02,wh-glx-a05u08 --pty /bin/bash
#
#   2. Run this script:
#      ./slurm/reset_devices.sh
# =============================================================================

set -e

if [ -z "$SLURM_JOB_NODELIST" ]; then
    echo "ERROR: Not running inside a slurm allocation."
    exit 1
fi

HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | paste -sd,)
NUM_HOSTS=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)

echo "Resetting devices on ${NUM_HOSTS} hosts: ${HOSTS}"

# Create temporary hostfile
HOSTFILE="/tmp/reset_hosts_$$.txt"
scontrol show hostnames $SLURM_JOB_NODELIST > ${HOSTFILE}

# Reset devices and clear shared memory
mpirun -np ${NUM_HOSTS} --hostfile ${HOSTFILE} --map-by ppr:1:node \
    bash -c '
        echo "=== $(hostname) ==="
        tt-smi -r 2>/dev/null || echo "tt-smi reset failed"
        sudo rm -rf /dev/shm/* 2>/dev/null || echo "shm clear skipped"
        echo "Done"
    '

rm -f ${HOSTFILE}
echo ""
echo "Reset complete on all nodes"
