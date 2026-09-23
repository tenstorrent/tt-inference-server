#!/bin/bash
set -euo pipefail
cd /home/ttuser/workspace/tt-inference-server-gemma/tt-media-server
export PYTHONPATH=/home/ttuser/workspace/gemma-reference/python
exec /home/ttuser/workspace/tt-inference-server/.venv-ttlab/bin/python ../deploy/tt-lab-gemma/start.py
