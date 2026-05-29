# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


class GenerateHelmValuesError(RuntimeError):
    """Raised when generate_helm_values cannot produce a valid values.yaml.

    Covers all upstream-data problems the CLI surfaces back to the operator:
    duplicate (model, device, engine, impl) tuples, multi-impl groups without
    a unique default_impl=True, missing engine info, etc.
    """
