#!/bin/bash
set -euo pipefail
echo "Reseting environment..."
unset TT_METAL_HOME
unset PYTHONPATH
unset WH_ARCH_YAML
unset ARCH_NAME
exec "$@"