# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
"""SGLang TT-Metal Plugin."""

from .patching.patching_model_registry import register_tt_models

__version__ = "0.1.0"

register_tt_models()  # patch SGLang ModelRegistry on import
