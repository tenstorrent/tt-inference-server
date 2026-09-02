# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

import pytest

from scripts.release.run_gemma_s4096_local_swe_gate import resolve_token_budget


def test_resolve_token_budget_preserves_exact_context_envelope():
    assert resolve_token_budget(32768, 2048) == (32768, 30720, 2048)


@pytest.mark.parametrize("context,output", [(0, 1), (4096, 0), (4096, 4096)])
def test_resolve_token_budget_rejects_invalid_envelopes(context, output):
    with pytest.raises(ValueError):
        resolve_token_budget(context, output)
