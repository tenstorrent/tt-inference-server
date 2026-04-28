#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Run the full AIME24 benchmark (all 30 questions) against a remote DeepSeek
# server exposing OpenAI-compatible endpoints.
#
# See helper_aime24_external.sh for required env vars.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "${HERE}/helper_aime24_external.sh" r1_aime24
