# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Recover an agent's in-container edit when the mini-swe-agent SWE-bench runner
crashes before it submits.

The pinned upstream runner (``mini-swe-agent==2.2.8``,
``minisweagent/run/benchmarks/swebench.py``) force-sets the prediction to the
empty string on *any* exception in ``process_instance``::

    except Exception as e:
        ...
        exit_status, result = type(e).__name__, ""

Because ``get_sb_environment`` builds ``--rm`` containers, the still-running
container that holds the agent's real, already-applied ``git diff`` is torn down
right after, so completed work is discarded and scored as an empty patch. This
is what threw away Qwen3.6-27B's correct edit at turn 71 and GPT-OSS-120B's at
turn 4.

The fix belongs in that shared runner (below every model and both ttis model
classes -- litellm and the token-budget class -- so it benefits all equally),
but the runner is a pinned third-party dependency. Rather than fork it or edit
site-packages on a node (an uncommitted, unbankable change), this module wraps
two stable seams of the runner from committed ttis code and is injected into the
agent subprocess via a generated ``sitecustomize`` on ``PYTHONPATH`` -- the same
mechanism ttis already uses for its SWE-bench harness patch
(``llm_module.agentic.swebench._add_mini_swe_capture_patch_to_env``):

* ``get_sb_environment(config, instance)`` -- remember the created environment,
  keyed by ``instance_id``.
* ``update_preds_file(path, instance_id, model_name, result)`` -- runs inside
  ``process_instance``'s ``finally`` while the container is still alive (the
  runner never calls ``env.cleanup``; teardown is deferred to ``__del__``). When
  ``result`` is empty (the discarded-crash case) and a diff can be recovered
  from the remembered container, substitute it before the prediction is written.

The wrappers rewrite the runner module's globals, so ``process_instance``'s bare
``get_sb_environment(...)`` / ``update_preds_file(...)`` calls resolve to them at
call time. Recovery never raises and never masks the original crash.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Mirrors how the mini-swe-agent SWE-bench config computes a submission diff
# (repo working dir is ``/testbed``, which is the environment's configured cwd,
# so ``env.execute`` runs there). ``git add -A`` also captures newly created
# files so the recovered patch is applyable by the verifier.
_RECOVER_COMMAND = "git add -A && git diff --cached"

_REGISTRY_LOCK = threading.Lock()
_ENV_BY_INSTANCE: dict[str, Any] = {}


def recover_patch_from_env(env: Any) -> str:
    """Best-effort snapshot of a live SWE-bench container's working-tree diff.

    Returns ``""`` when there is no environment or nothing to recover. Never
    raises: recovery must not mask the original crash.
    """
    if env is None:
        return ""
    try:
        out = env.execute({"command": _RECOVER_COMMAND})
    except Exception:  # noqa: BLE001 -- recovery is best-effort, never fatal
        logger.warning(
            "mini-swe capture: env.execute failed during patch recovery",
            exc_info=True,
        )
        return ""
    if not isinstance(out, dict) or out.get("returncode") != 0:
        return ""
    output = out.get("output")
    return output if isinstance(output, str) else ""


def register_environment(instance_id: str, env: Any) -> None:
    if not instance_id:
        return
    with _REGISTRY_LOCK:
        _ENV_BY_INSTANCE[instance_id] = env


def pop_environment(instance_id: str) -> Any:
    with _REGISTRY_LOCK:
        return _ENV_BY_INSTANCE.pop(instance_id, None)


def make_get_sb_environment_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``get_sb_environment`` to remember each instance's live environment."""

    def wrapper(config: dict, instance: dict) -> Any:
        env = original(config, instance)
        try:
            register_environment((instance or {}).get("instance_id", ""), env)
        except Exception:  # noqa: BLE001 -- registration must never break a run
            logger.warning(
                "mini-swe capture: failed to register environment", exc_info=True
            )
        return env

    wrapper._ttis_capture_wrapped = True  # type: ignore[attr-defined]
    return wrapper


def make_update_preds_file_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``update_preds_file`` to backfill an empty crash result with the
    recovered in-container diff before it is persisted."""

    def wrapper(
        output_path: Any, instance_id: str, model_name: str, result: str
    ) -> Any:
        env = pop_environment(instance_id)
        if not result and env is not None:
            recovered = recover_patch_from_env(env)
            if recovered.strip():
                logger.warning(
                    "mini-swe capture: recovered a %d-char in-container patch for "
                    "%s that the runner would otherwise discard as an empty "
                    "prediction on crash",
                    len(recovered),
                    instance_id,
                )
                result = recovered
        return original(output_path, instance_id, model_name, result)

    wrapper._ttis_capture_wrapped = True  # type: ignore[attr-defined]
    return wrapper


_INSTALLED = False


def install() -> bool:
    """Idempotently wrap the pinned mini-swe-agent SWE-bench runner seams.

    Returns ``True`` once the wrappers are in place, ``False`` if the runner
    cannot be imported (recovery is then simply inactive -- never fatal).
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    try:
        from minisweagent.run.benchmarks import swebench as runner
    except Exception:  # noqa: BLE001 -- absence of the dep must not break import
        logger.warning(
            "mini-swe capture: minisweagent runner not importable; capture inactive",
            exc_info=True,
        )
        return False
    if not getattr(runner.get_sb_environment, "_ttis_capture_wrapped", False):
        runner.get_sb_environment = make_get_sb_environment_wrapper(
            runner.get_sb_environment
        )
    if not getattr(runner.update_preds_file, "_ttis_capture_wrapped", False):
        runner.update_preds_file = make_update_preds_file_wrapper(
            runner.update_preds_file
        )
    _INSTALLED = True
    return True
