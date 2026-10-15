# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Harbor datasets generated on the host instead of pulled from the registry.

A registry copy can lag behind the adapter that produced it, for example when
the adapter has since fixed its task Dockerfiles. A dataset with an entry in
``_GENERATORS`` is then built from the adapter in the pinned Harbor checkout
and handed to Harbor as a local path. Generated tasks keep the registry task
names in ``task.toml``, but Harbor filters a local dataset by directory name,
so configured task-name patterns are translated with
:func:`local_task_pattern`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

# Registry dataset name -> generator. A generator receives the EVALS_AGENTIC
# interpreter, whose editable Harbor install carries the adapters, and returns
# the directory of generated tasks.
_GENERATORS: Dict[str, Callable[[Path], Path]] = {}


def local_task_pattern(dataset: str, pattern: str) -> str:
    """Map a registry task-name pattern onto the generated directory names.

    ``sierra-research/tau3-bench__tau3-banking_knowledge-*`` becomes
    ``tau3-banking_knowledge-*``; patterns without the prefix pass through.
    """
    prefix = f"{dataset}__"
    return pattern[len(prefix) :] if pattern.startswith(prefix) else pattern


def prepare_local_dataset(dataset: str, python: Path) -> Optional[Path]:
    """Task directory for *dataset* when it is generated locally, else ``None``.

    *python* is the EVALS_AGENTIC interpreter. Datasets without a generator are
    resolved from the registry as before.
    """
    generator = _GENERATORS.get(dataset)
    if generator is None:
        return None
    return generator(Path(python))
