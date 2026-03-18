#!/bin/bash

# Script to generate an empty report from template using envsubst
# Usage: generate-empty-report.sh <output_file> <job_id> <model> <device> <workflow> [tt_metal_commit] [vllm_commit]

set -e

OUTPUT_FILE="$1"
export JOB_ID="$2"
export MODEL="$3"
export DEVICE="$4"
export WORKFLOW="$5"
export TT_METAL_COMMIT="${6:-}"
export VLLM_COMMIT="${7:-}"

if [ $# -lt 5 ]; then
    echo "Usage: $0 <output_file> <job_id> <model> <device> <workflow> [tt_metal_commit] [vllm_commit]"
    exit 1
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE_FILE="$SCRIPT_DIR/templates/empty-report-template.json"

# Check if template exists
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "Error: Template file not found at $TEMPLATE_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Use envsubst to substitute environment variables
envsubst < "$TEMPLATE_FILE" > "$OUTPUT_FILE"

echo "Empty report generated: $OUTPUT_FILE"
