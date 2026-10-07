#!/bin/bash
# Gemma 4 31B on four Blackhole cards. Usage: start.sh [ENV_FILE] (default: server.env, one
# sequence at a time; server-batch.env batches eight).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
cd "$here/../../tt-media-server"
# The tt-lab deployments' isolated environment (read-only here), and the Gemma 4 tokenizer support.
venv=${TT_LAB_VENV:-/home/ttuser/workspace/tt-inference-server/.venv-ttlab}
export PYTHONPATH=/home/ttuser/workspace/gemma-reference/python
exec "$venv/bin/python" ../deploy/tt-lab/start.py "${1:-$here/server.env}"
