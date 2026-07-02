# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the no-device dispatch tests (issue #46).

These tests must NEVER touch a Tenstorrent device. The dispatch package only
imports ttnn function-locally, so importing dispatcher/runner is safe — but a
buggy test could still call into a code path that opens the device. To make that
impossible, we replace `ttnn` in sys.modules with a stub whose open_device (and
every other attribute) raises. Any test that accidentally tries to open the
device fails loudly instead of hanging on missing hardware.

This conftest's effect is scoped to the tests/dispatch/ directory — and the CI
workflow runs only this directory — so the stub never leaks into device-bound
runs of the wider suite.
"""

import sys
import types


class _ForbiddenTtnn(types.ModuleType):
    """A fake ttnn module: any attribute access raises (no device in CI)."""

    def __getattr__(self, name):  # noqa: D401 - simple guard
        # Let dunders (__file__, __spec__, __path__, ...) behave as ABSENT so the
        # interpreter's own introspection (inspect/import machinery, which torch's
        # lazy op registration walks during `import torch`) doesn't trip over a
        # callable where it expects a path/None.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _hard_fail(*args, **kwargs):
            raise RuntimeError(
                f"ttnn.{name} was called in a no-device CI test — forbidden. "
                "These tests must run with no Tenstorrent card. If you need the "
                "device, write the test in the device-bound harness under ~/dispatch/tests/."
            )

        # open_device specifically must hard-fail; return a raiser for everything else too.
        return _hard_fail


# Install the stub unconditionally so it shadows any real ttnn install too — these
# tests are device-free by contract.
sys.modules["ttnn"] = _ForbiddenTtnn("ttnn")
