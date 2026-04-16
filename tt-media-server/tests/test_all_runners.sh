#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# Comprehensive test script for all 4 video runner configurations
# Run this from tt-media-server directory

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Generate unique SHM names
TIMESTAMP=$(date +%s)
export TT_IPC_SHM_VIDEO_REQ="tt_video_req_${TIMESTAMP}"
export TT_IPC_SHM_VIDEO_RESP="tt_video_resp_${TIMESTAMP}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing All 4 Video Runner Configurations${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Shared Memory Names:"
echo -e "  Request:  ${GREEN}${TT_IPC_SHM_VIDEO_REQ}${NC}"
echo -e "  Response: ${GREEN}${TT_IPC_SHM_VIDEO_RESP}${NC}"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"

    # Kill any running video_runner.py processes
    pkill -f "python.*video_runner.py" || true

    # Remove shared memory segments
    rm -f "/dev/shm/${TT_IPC_SHM_VIDEO_REQ}" || true
    rm -f "/dev/shm/${TT_IPC_SHM_VIDEO_RESP}" || true

    echo -e "${GREEN}Cleanup complete${NC}"
}

trap cleanup EXIT

# Function to test multi-rank configuration
test_multi_rank() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Multi-Rank Test (All 4 ranks)${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""

    export MODEL_RUNNER=tt_mochi_1

    # Start ranks 1-3 first (workers)
    echo -e "${YELLOW}Starting worker ranks...${NC}"

    export RANK=1
    export TT_VISIBLE_DEVICES=1
    python tt_model_runners/video_runner.py &
    RANK1_PID=$!
    echo -e "  Rank 1 started (PID: ${RANK1_PID})"

    export RANK=2
    export TT_VISIBLE_DEVICES=2
    python tt_model_runners/video_runner.py &
    RANK2_PID=$!
    echo -e "  Rank 2 started (PID: ${RANK2_PID})"

    export RANK=3
    export TT_VISIBLE_DEVICES=3
    python tt_model_runners/video_runner.py &
    RANK3_PID=$!
    echo -e "  Rank 3 started (PID: ${RANK3_PID})"

    # Wait for workers to start listening
    sleep 3

    # Start rank 0 (coordinator)
    echo -e "${YELLOW}Starting rank 0 coordinator...${NC}"
    export RANK=0
    export TT_VISIBLE_DEVICES=0
    python tt_model_runners/video_runner.py &
    RANK0_PID=$!
    echo -e "  Rank 0 started (PID: ${RANK0_PID})"

    # Wait for initialization
    sleep 5

    # Send test requests
    echo -e "${YELLOW}Sending test requests...${NC}"
    python tt_model_runners/test_video_shm.py

    # Give time to process and distribute
    sleep 5

    # Kill all ranks
    echo -e "${YELLOW}Stopping all ranks...${NC}"
    kill $RANK0_PID $RANK1_PID $RANK2_PID $RANK3_PID || true
    wait $RANK0_PID $RANK1_PID $RANK2_PID $RANK3_PID 2>/dev/null || true

    echo -e "${GREEN}✓ Multi-rank test complete${NC}"
}

# Main execution
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Multi-Rank Video Runner Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

test_multi_rank

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test completed!${NC}"
echo -e "${GREEN}========================================${NC}"

