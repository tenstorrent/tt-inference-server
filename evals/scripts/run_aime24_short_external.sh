#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# Run the 5-question short AIME24 subset (IDs 60, 69, 75, 84, 86 — the lowest
# average GPU token counts per
# https://github.com/tenstorrent/tt-metal/issues/37857#issuecomment-4116812760)
# against a remote DeepSeek server.
#
# See helper_aime24_external.sh for required env vars.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

exec "${HERE}/helper_aime24_external.sh" \
    r1_aime24_short \
    "${REPO_ROOT}/evals/custom_tasks/r1_aime24_short"
