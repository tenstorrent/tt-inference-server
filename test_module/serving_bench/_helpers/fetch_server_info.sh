#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# fetch_server_info.sh <target_url>
#
# GET <target>/info and print the JSON body to stdout. Falls back to '{}' on
# network failure or invalid JSON so callers can always trust the output is
# valid JSON.

set -Eeuo pipefail

TARGET="${1:?usage: fetch_server_info.sh <target_url>}"

INFO="$(python3 -c "import sys, urllib.request; print(urllib.request.urlopen(sys.argv[1] + '/info', timeout=10).read().decode())" "$TARGET" 2>/dev/null || true)"

if [ -z "$INFO" ] || ! echo "$INFO" | jq empty 2>/dev/null; then
  echo "warning: /info returned invalid or empty JSON; falling back to {}" >&2
  echo '{}'
else
  echo "$INFO"
fi
