#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# wait_for_server.sh <target_url>
#
# Poll <target>/tt-liveness every 10s until it responds 200. Used by the
# dispatcher before invoking any test script.

set -Eeuo pipefail

TARGET="${1:?usage: wait_for_server.sh <target_url>}"

echo "==> Waiting for $TARGET/tt-liveness ..."
until python3 -c "import sys, urllib.request; urllib.request.urlopen(sys.argv[1] + '/tt-liveness', timeout=5)" "$TARGET" 2>/dev/null; do
  echo "    not ready yet, retrying in 10s..."
  sleep 10
done
echo "==> Server is live."
