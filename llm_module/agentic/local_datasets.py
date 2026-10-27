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

Running a local copy instead of the registry dataset also changes:

- Scoring: Harbor reports the job metrics the verifier writes, or a plain
  mean of the rewards; a registry dataset's published metric does not apply.
- ``n_tasks``: Harbor truncates a local dataset in sorted directory order,
  not registry order, so a capped run can pick different tasks.
- Provenance: the job lock still records each task's content digest,
  computed from the generated directory, but there is no registry
  ``name@ref`` to cite.

A generator must therefore emit the same tasks, under the registry's task
names, as the registry copy it replaces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

# Eval-config dataset name -> generator. A generator receives the EVALS_AGENTIC
# interpreter, whose editable Harbor install carries the adapters, and returns
# the directory of generated tasks.
_GENERATORS: Dict[str, Callable[[Path], Path]] = {}


def local_task_pattern(dataset: str, pattern: str) -> str:
    """Map a registry task-name pattern onto the generated directory names.

    ``sierra-research/tau3-bench__tau3-banking_knowledge-*`` becomes
    ``tau3-banking_knowledge-*``; patterns without the prefix pass through.
    Local directory names cannot contain ``/``, so a pattern that still does
    (another dataset's prefix, a typo) raises: Harbor would ignore an
    unmatched exclude pattern without a word.
    """
    prefix = f"{dataset}__"
    local = pattern[len(prefix) :] if pattern.startswith(prefix) else pattern
    if "/" in local:
        raise ValueError(
            f"Task pattern {pattern!r} cannot match a local copy of {dataset!r}"
        )
    return local


def prepare_local_dataset(dataset: str, python: Path) -> Optional[Path]:
    """Task directory for *dataset* when it is generated locally, else ``None``.

    *python* is the EVALS_AGENTIC interpreter. Datasets without a generator are
    resolved from the registry as before.
    """
    generator = _GENERATORS.get(dataset)
    if generator is None:
        return None
    return generator(Path(python))
