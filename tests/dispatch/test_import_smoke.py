# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""No-device import smoke test (issue #46).

The dispatch package imports ttnn only function-locally, so importing the modules
and referencing the top-level classes must succeed with NO device present. conftest
installs a hard-failing ttnn stub, so if any of these imports tried to touch the
device the test would error loudly rather than pass by accident.
"""


def test_import_dispatcher():
    from tt_inference_server.dispatch.dispatcher import (  # noqa: F401
        Capabilities,
        KernelDispatcher,
        ModelMatrixEntry,
        derive_capabilities,
    )

    assert KernelDispatcher is not None
    assert callable(derive_capabilities)


def test_import_runner():
    from tt_inference_server.dispatch.runner import TTModelRunner  # noqa: F401

    assert TTModelRunner is not None


def test_import_validator_and_public_api():
    from tt_inference_server.dispatch import load_model, ModelHandle  # noqa: F401
    from tt_inference_server.dispatch.output_validator import validate_output

    assert callable(load_model)
    assert callable(validate_output)


def test_ttnn_is_the_forbidden_stub():
    """Belt-and-suspenders: confirm conftest's guard is active for this dir."""
    import ttnn
    import pytest

    with pytest.raises(RuntimeError):
        ttnn.open_device(device_id=0)
