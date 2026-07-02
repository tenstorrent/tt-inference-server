"""Single-agent loop: run a persona until it produces a text response."""

import json
import re
import time
from datetime import datetime, timezone
import openai
from openai import OpenAI
from orchestrator.config import PROVIDER_REGISTRY, get_api_key
import orchestrator.tools as T

_BACKOFF_SECONDS = [5, 15, 30, 60, 120, 300, 300, 300, 300, 300]

# Default hard cap on tool-call iterations.  This is a safety rail, not a
# cost-control mechanism — cost is better managed at the token/dollar level.
# Callers can lower it for simple tasks or raise it for complex ones by
# passing max_tool_rounds explicitly to run().
DEFAULT_MAX_TOOL_ROUNDS = 100

# Tools the orchestrator calls directly; agents must never invoke them.
_ORCHESTRATOR_ONLY = frozenset({"create_pr"})

# Matches complete <think>...</think> blocks and any bare </think> closers
# left behind by reasoning models that emit unclosed tags.
_THINK_RE = re.compile(r"<think>.*?</think>|</think>", re.DOTALL)

# Prefix used when injecting prior reasoning into history so orchestrator.py
# can locate and extract the most-recent reasoning block from a returned history.
REASONING_INJECTION_PREFIX = "[System: Your reasoning from the previous turn:]"

_PR_BODY_SYSTEM = (
    "You are a technical writer producing a GitHub pull request description. "
    "Output ONLY the PR body — no preamble, no commentary, no markdown code fences. "
    "Use exactly these four sections in this order:\n\n"
    "## Summary\n"
    "<one short paragraph: what this PR does and why>\n\n"
    "## Changes\n"
    "<bullet list of files/components changed and what changed in each>\n\n"
    "## Testing\n"
    "<how the change was tested; tests added or updated>\n\n"
    "## Fixes\n"
    "Fixes #N   ← use the issue number from the task, or write N/A if there is none"
)

_PR_BODY_USER_TEMPLATE = """\
Task description:
{task}

git diff (main..HEAD):
{diff}

Write the PR body now."""


def _strip_think(text: str) -> str:
    # Replace with a space so adjacent words around a mid-sentence block aren't concatenated.
    return _THINK_RE.sub(" ", text).strip()


# Truncate reasoning to keep context growth bounded; 8 k chars covers all
# realistic plans while preventing a runaway model from filling the window.
_MAX_REASONING_CHARS = 8_000

def _make_reasoning_message(reasoning_content: str) -> dict:
    # Coerce to str defensively — some provider SDKs return non-string objects.
    safe = str(reasoning_content)[:_MAX_REASONING_CHARS]
    body = (
        f"{REASONING_INJECTION_PREFIX}\n"
        f"<reasoning>\n{safe}\n</reasoning>\n"
        "Use this as your plan. Proceed with implementation."
    )
    return {"role": "system", "content": body}


class MaxToolRoundsError(Exception):
    """Raised when an agent exhausts its max_tool_rounds budget without
    producing a final text response.

    This is an orchestrator-level failure: it means the agent's work is
    incomplete.  Callers must NOT pass the partial result to downstream
    agents (e.g. reviewers) as if it were finished work.
    """

    def __init__(self, persona_name: str, max_tool_rounds: int, history: list[dict]):
        self.persona_name = persona_name
        self.max_tool_rounds = max_tool_rounds
        self.history = history
        super().__init__(
            f"ERROR: {persona_name} hit max_tool_rounds ({max_tool_rounds}) without finishing"
        )


def _client(provider: str, api_key: str | None = None) -> OpenAI:
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider {provider!r}. Valid providers: {sorted(PROVIDER_REGISTRY)}"
        )
    entry = PROVIDER_REGISTRY[provider]
    if provider == "litellm":
        key = get_api_key(api_key)
    else:
        # Non-litellm providers use their own key resolution; api_key override
        # is not supported because the CLI --api-key flag targets LiteLLM only.
        key = entry["get_key"]()
    return OpenAI(base_url=entry["base_url"], api_key=key)


def generate_pr_body(
    implementer: dict,
    task: str,
    diff: str,
    api_key: str | None = None,
) -> str:
    # No tools — model must produce the body text directly.
    # Must use the implementer's model+provider, not a hardcoded fallback.
    provider = implementer.get("provider", "litellm")
    client = _client(provider, api_key)

    # Diff can be very large; cap it so we don't blow past context limits.
    diff_truncated = diff[:12_000] + ("\n[diff truncated]" if len(diff) > 12_000 else "")

    messages = [
        {"role": "system", "content": _PR_BODY_SYSTEM},
        {
            "role": "user",
            "content": _PR_BODY_USER_TEMPLATE.format(task=task, diff=diff_truncated),
        },
    ]

    kwargs: dict = dict(model=implementer["model"], messages=messages)
    if "max_tokens" in implementer:
        kwargs["max_tokens"] = implementer["max_tokens"]

    _max_attempts = len(_BACKOFF_SECONDS) + 1
    for attempt in range(_max_attempts):
        try:
            response = client.chat.completions.create(**kwargs)
            text = _strip_think(response.choices[0].message.content or "")
            return text.strip()
        except openai.RateLimitError as e:
            if attempt >= len(_BACKOFF_SECONDS):
                raise
            reset_time = None
            m = re.search(r"Limit resets at:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+)", str(e))
            if m:
                try:
                    reset_time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                except ValueError:
                    reset_time = None
            if reset_time is not None:
                sleep_secs = max(0, (reset_time - datetime.now(timezone.utc)).total_seconds()) + 5
                time.sleep(sleep_secs)
            else:
                time.sleep(_BACKOFF_SECONDS[attempt])
        except (openai.InternalServerError, openai.APIConnectionError):
            if attempt >= len(_BACKOFF_SECONDS):
                raise
            time.sleep(_BACKOFF_SECONDS[attempt])

    raise RuntimeError("generate_pr_body: retry loop exited without response")


def run(
    persona: dict,
    messages: list[dict],
    cwd: str | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    verbose: bool = True,
    api_key: str | None = None,
    exclude_tools: set[str] | None = None,
    inject_reasoning: bool = False,
    prior_reasoning: str | None = None,
) -> tuple[str, list[dict]]:
    provider = persona.get("provider", "litellm")
    client = _client(provider, api_key)
    history = [{"role": "system", "content": persona["system"]}] + messages

    # Prepend prior-turn reasoning so the model can resume from its plan.
    if inject_reasoning and prior_reasoning:
        history.append(_make_reasoning_message(prior_reasoning))

    blocked = _ORCHESTRATOR_ONLY | (exclude_tools or set())
    agent_tools = [t for t in T.DEFS if t["function"]["name"] not in blocked]

    _max_attempts = len(_BACKOFF_SECONDS) + 1

    for round_num in range(max_tool_rounds):
        kwargs = dict(
            model=persona["model"],
            messages=history,
            tools=agent_tools,
            tool_choice="auto",
        )
        if "max_tokens" in persona:
            kwargs["max_tokens"] = persona["max_tokens"]
        _attempt_start = time.monotonic()
        for attempt in range(_max_attempts):
            try:
                response = client.chat.completions.create(**kwargs)
                break
            except openai.RateLimitError as e:
                if attempt >= len(_BACKOFF_SECONDS):
                    raise
                reset_time = None
                m = re.search(r"Limit resets at:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \w+)", str(e))
                if m:
                    try:
                        reset_time = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S %Z").replace(tzinfo=timezone.utc)
                    except ValueError:
                        reset_time = None
                if reset_time is not None:
                    sleep_secs = max(0, (reset_time - datetime.now(timezone.utc)).total_seconds()) + 5
                    if verbose:
                        elapsed = time.monotonic() - _attempt_start
                        print(
                            f"  [{persona['name']}] rate limit (attempt {attempt + 1}/{_max_attempts},"
                            f" elapsed {elapsed:.0f}s):"
                            f" reset at {reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')};"
                            f" sleeping {sleep_secs:.1f}s"
                        )
                    time.sleep(sleep_secs)
                else:
                    next_wait = _BACKOFF_SECONDS[attempt]
                    if verbose:
                        elapsed = time.monotonic() - _attempt_start
                        print(
                            f"  [{persona['name']}] rate limit (attempt {attempt + 1}/{_max_attempts},"
                            f" elapsed {elapsed:.0f}s):"
                            f" unknown reset time (using backoff); retrying in {next_wait}s"
                        )
                    time.sleep(next_wait)
            except (openai.InternalServerError, openai.APIConnectionError) as e:
                if attempt >= len(_BACKOFF_SECONDS):
                    raise
                next_wait = _BACKOFF_SECONDS[attempt]
                if verbose:
                    elapsed = time.monotonic() - _attempt_start
                    # Log only the exception type — the full str(e) may contain
                    # raw HTTP response bodies with internal infrastructure details.
                    print(
                        f"  [{persona['name']}] transient error (attempt {attempt + 1}/{_max_attempts},"
                        f" elapsed {elapsed:.0f}s):"
                        f" {type(e).__name__}; retrying in {next_wait}s"
                    )
                time.sleep(next_wait)
        else:
            raise RuntimeError("retry loop exited without response")
        msg = response.choices[0].message

        content = _strip_think(msg.content or "")

        # Skip pure-reasoning responses without adding anything to history;
        # the model hasn't actually responded yet.
        if not msg.tool_calls and not content:
            continue

        # Pre-scan arguments for parse errors before writing history. We must
        # store "{}" for any malformed entry so downstream APIs (e.g. Anthropic
        # via litellm) don't reject the history with a 400 (see #146). We keep
        # the exception so the exact parse error can still be fed back to the
        # model (preserving #133 behaviour).
        parse_errors: dict[str, json.JSONDecodeError] = {}
        for tc in (msg.tool_calls or []):
            try:
                json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                parse_errors[tc.id] = exc

        # Append assistant turn (convert to dict safely)
        assistant_entry = {"role": "assistant", "content": content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        # Sanitize malformed arguments so history stays valid JSON.
                        "arguments": "{}" if tc.id in parse_errors else tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        history.append(assistant_entry)

        if not msg.tool_calls:
            return content, history

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            name = tc.function.name
            if tc.id in parse_errors:
                # Surface malformed JSON back to the model so it can retry
                # rather than crashing the entire run (see #133).
                error_msg = f"ERROR: could not parse tool arguments as JSON: {parse_errors[tc.id]}"
                if verbose:
                    print(f"  [{persona['name']}] {error_msg}")
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": error_msg,
                })
                continue
            args = json.loads(tc.function.arguments)
            if verbose:
                print(f"  [{persona['name']}] tool: {name}({list(args.keys())})")
            result = T.execute(name, args, cwd=cwd)
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

        # Carry forward any reasoning the model emitted so it can resume its plan
        # on the next tool-call iteration without re-deriving it from scratch.
        if inject_reasoning:
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning and isinstance(reasoning, (str, bytes, int, float)):
                history.append(_make_reasoning_message(reasoning))

    raise MaxToolRoundsError(persona["name"], max_tool_rounds, history)
