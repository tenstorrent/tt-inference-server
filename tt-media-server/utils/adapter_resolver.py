# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from utils.adapter_storage import AdapterInfo, AdapterStorage, get_adapter_storage

# Re-export so existing imports continue to work.
__all__ = ["AdapterInfo", "resolve_adapter"]


def resolve_adapter(
    adapter: str, storage: AdapterStorage | None = None
) -> AdapterInfo:
    """Resolve an adapter identifier to base model name + loadable path.

    Args:
        adapter: Adapter reference in the format "{job_id}/{checkpoint_id}",
                 e.g. "110aa287-8607-4d82-814e-69492b55a4e1/ckpt-step-20".
        storage: Optional explicit storage backend.  When *None* the default
                 backend is derived from application settings.

    Returns:
        AdapterInfo with the base model name and the adapter path.
    """
    if storage is None:
        storage = get_adapter_storage()
    return storage.resolve_adapter(adapter)
