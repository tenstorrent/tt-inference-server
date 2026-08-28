"""mini-swe-agent model class that bounds and records the format-error loop.

This module is not imported by tt-inference-server itself. It is placed on the
agent subprocess's ``PYTHONPATH`` and referenced by class path from the
generated ``mini_sweagent_model_config.yaml``, so it runs inside the agentic
venv where ``minisweagent`` is installed.

Why it exists: ``LitellmModel.query()`` builds ``message["extra"]["actions"]``
by calling ``_parse_actions()``, which raises ``FormatError`` when the response
carries no tool call. That happens *before* the agent appends the assistant
message, so the malformed response is discarded -- it never reaches the
trajectory and the model never sees what it did wrong. A model that cannot emit
a tool call therefore keeps failing the same way until ``step_limit``, spending
one full ``max_tokens`` generation per attempt, and leaves no evidence behind.

So this class does two things: it caps the streak, and it saves each discarded
response to disk first.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from minisweagent.exceptions import FormatError, LimitsExceeded
from minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig

# Must live under the "minisweagent" namespace: that logger owns the RichHandler
# that reaches the terminal and the FileHandler that writes minisweagent.log, and
# it is the one set to DEBUG. A top-level logger name would fall back to
# logging.lastResort -- bare stderr output, nothing in the log file.
logger = logging.getLogger("minisweagent.tt_format_guard")

DEFAULT_MAX_CONSECUTIVE_FORMAT_ERRORS = 10
# Tool observations (a stray `find /` or `cat` of a big file) dominate a prompt
# and are rarely what explains the model losing the tool-call format, so clip
# each recorded message rather than letting one dump reach hundreds of MB.
DEFAULT_MAX_MESSAGE_CHARS = 10_000
FORMAT_ERROR_LOOP_EXIT_STATUS = "FormatErrorLoop"

_UNSAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_UNKNOWN_INSTANCE = "unknown_instance"

# Set by the wrapper installed below; read when labelling a dump.
_current_instance = threading.local()


def _install_instance_id_tracker() -> None:
    """Label dumps with the SWE-bench instance the worker thread is running.

    The model is built per instance but is never told which one, the batch
    runner keeps its started set only in memory, and the instance template
    exposes just ``{{task}}``. The runner's environment factory does receive the
    instance dict and is called after ``get_model()`` on the same thread, so
    wrapping it is the cheapest reliable place to capture the id.

    Best-effort: a failure here only costs us the label, so never raise.
    """
    try:
        from minisweagent.run.benchmarks import swebench as runner
    except Exception:  # noqa: BLE001 - non-swebench entrypoints have no runner
        return
    original = getattr(runner, "get_sb_environment", None)
    if original is None or getattr(original, "_tt_tracks_instance_id", False):
        return

    def tracked(config, instance):
        try:
            _current_instance.instance_id = instance.get("instance_id")
        except Exception:  # noqa: BLE001
            pass
        return original(config, instance)

    tracked._tt_tracks_instance_id = True
    runner.get_sb_environment = tracked


_install_instance_id_tracker()


class FormatErrorGuardConfig(LitellmModelConfig):
    max_consecutive_format_errors: int = DEFAULT_MAX_CONSECUTIVE_FORMAT_ERRORS
    """Abort the instance after this many back-to-back format errors. 0 disables."""
    format_error_dump_dir: Optional[str] = None
    """Directory for one JSON file per discarded response. None disables saving."""
    format_error_max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS
    """Per-message cap when recording the prompt. 0 keeps every message whole."""


class FormatErrorGuardModel(LitellmModel):
    """``LitellmModel`` that records discarded responses and gives up on a loop.

    The counter resets on every successfully parsed response, so a model that
    recovers on its own is never penalised.
    """

    def __init__(self, **kwargs):
        super().__init__(config_class=FormatErrorGuardConfig, **kwargs)
        self.consecutive_format_errors = 0
        self.format_errors_seen = 0
        self.last_messages: list[dict] = []

    def query(self, messages: list[dict], **kwargs) -> dict:
        """Remember the prompt so a dump can record what produced the response.

        ``_parse_actions`` is handed only the response, and one model instance
        serves one SWE-bench instance on one thread, so stashing it here is safe.
        """
        self.last_messages = list(messages or ())
        return super().query(messages, **kwargs)

    # ----------------------------------------------------------------- dumps

    def _record_prompt(self, messages: list[dict]) -> tuple[list[dict], int]:
        """Flatten the sent conversation for the dump. Returns (messages, chars).

        Only the fields that explain the model's behaviour are kept. In
        particular ``extra`` is dropped: it carries the entire raw API response
        of every previous turn, which would multiply the dump size by the number
        of steps taken so far.
        """
        cap = self.config.format_error_max_message_chars
        recorded: list[dict[str, Any]] = []
        total_chars = 0

        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            text = (
                content
                if isinstance(content, str)
                else json.dumps(content, default=str)
            )
            total_chars += len(text)

            entry: dict[str, Any] = {"role": message.get("role")}
            if cap > 0 and len(text) > cap:
                entry["content"] = text[:cap]
                entry["content_truncated_from"] = len(text)
            else:
                entry["content"] = text
            # Never clipped: a half a JSON string cannot be parsed back, so the
            # replay tooling would lose the call entirely to save a few KB.
            if tool_calls := message.get("tool_calls"):
                entry["tool_calls"] = json.dumps(tool_calls, default=str)
            # The agent keeps the whole response message in its history, so prior
            # thought channels really are sent back on every turn.
            if reasoning := message.get("reasoning_content"):
                total_chars += len(reasoning)
                if cap > 0 and len(reasoning) > cap:
                    entry["reasoning_content"] = reasoning[:cap]
                    entry["reasoning_truncated_from"] = len(reasoning)
                else:
                    entry["reasoning_content"] = reasoning
            if tool_call_id := message.get("tool_call_id"):
                entry["tool_call_id"] = tool_call_id
            # Marks the messages the harness injected for earlier format errors.
            if interrupt := (message.get("extra") or {}).get("interrupt_type"):
                entry["interrupt_type"] = interrupt
            recorded.append(entry)

        return recorded, total_chars

    @staticmethod
    def _response_details(response: Any) -> dict[str, Any]:
        """Pull the interesting fields off a response without ever raising.

        ``finish_reason`` is the one to read first: ``"length"`` means the model
        burned its whole ``max_tokens`` budget on prose.
        """
        details: dict[str, Any] = {"model": getattr(response, "model", None)}
        try:
            choice = response.choices[0]
        except Exception:  # noqa: BLE001
            return details

        details["finish_reason"] = getattr(choice, "finish_reason", None)
        message = getattr(choice, "message", None)
        if message is not None:
            content = getattr(message, "content", None)
            details["content"] = content
            details["content_chars"] = len(content) if content else 0
            if reasoning := getattr(message, "reasoning_content", None):
                details["reasoning_content"] = reasoning
            tool_calls = getattr(message, "tool_calls", None)
            details["tool_calls"] = [str(tc) for tc in tool_calls or []]

        usage = getattr(response, "usage", None)
        if usage is not None:
            try:
                details["usage"] = usage.model_dump()
            except Exception:  # noqa: BLE001
                details["usage"] = str(usage)
        return details

    def _save_format_error(
        self, response: Any, error: FormatError, instance_id: str
    ) -> tuple[dict[str, Any], Optional[Path]]:
        """Write the discarded response to its own file. Returns (details, path)."""
        details = self._response_details(response)
        raw_dir = self.config.format_error_dump_dir
        if not raw_dir:
            return details, None

        # The nudge the harness sends back in place of the response.
        sent_back = [m.get("content") for m in getattr(error, "messages", ())]
        prompt, prompt_chars = self._record_prompt(self.last_messages)
        payload = {
            "instance_id": instance_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "consecutive_format_errors": self.consecutive_format_errors,
            "format_errors_seen": self.format_errors_seen,
            "model_name": self.config.model_name,
            "format_error_sent_to_model": sent_back,
            **details,
            "prompt_messages": len(prompt),
            "prompt_chars": prompt_chars,
            "prompt": prompt,
        }

        safe_id = _UNSAFE_FILENAME_RE.sub("_", instance_id)
        path = Path(raw_dir) / f"{safe_id}_{self.format_errors_seen:02d}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - never fail a run over a dump
            logger.warning("Could not save format-error dump to %s: %s", path, exc)
            return details, None
        return details, path

    # ----------------------------------------------------------------- hook

    def _parse_actions(self, response) -> list[dict]:
        try:
            actions = super()._parse_actions(response)
        except FormatError as error:
            self.consecutive_format_errors += 1
            self.format_errors_seen += 1
            instance_id = (
                getattr(_current_instance, "instance_id", None) or _UNKNOWN_INSTANCE
            )
            details, dump_path = self._save_format_error(response, error, instance_id)

            # The response text itself stays out of the log -- it runs to tens of
            # thousands of characters and repeats every attempt. Read the dump.
            logger.warning(
                "%s: response %s had no usable tool call (streak %s/%s, "
                "finish_reason=%s, %s response chars, %s prompt messages)%s",
                instance_id,
                self.format_errors_seen,
                self.consecutive_format_errors,
                self.config.max_consecutive_format_errors or "inf",
                details.get("finish_reason"),
                details.get("content_chars", 0),
                len(self.last_messages),
                f", saved to {dump_path}" if dump_path else "",
            )

            limit = self.config.max_consecutive_format_errors
            if limit > 0 and self.consecutive_format_errors >= limit:
                logger.warning(
                    "Aborting %s: %s consecutive responses could not be parsed "
                    "as a tool call (limit=%s).",
                    instance_id,
                    self.consecutive_format_errors,
                    limit,
                )
                # role="exit" is what DefaultAgent.run() breaks on, so this ends
                # the instance with an empty submission instead of looping to
                # step_limit. The runner still records it in preds.json.
                raise LimitsExceeded(
                    {
                        "role": "exit",
                        "content": FORMAT_ERROR_LOOP_EXIT_STATUS,
                        "extra": {
                            "exit_status": FORMAT_ERROR_LOOP_EXIT_STATUS,
                            "submission": "",
                        },
                    }
                ) from None
            raise
        self.consecutive_format_errors = 0
        return actions
