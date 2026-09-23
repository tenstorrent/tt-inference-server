#!/bin/bash
set -euo pipefail
cd /home/ttuser/workspace/tt-inference-server/tt-media-server
# Let systemd load server.env, or parse it without evaluating shell content.
exec ../.venv-ttlab/bin/python ../deploy/tt-lab/start.py
