"""Single-agent loop: run a persona until it produces a text response."""

import json
import re
import time
from datetime import datetime, timezone
import openai
from openai import OpenAI
from orchestrator.config import PROVIDER_REGISTRY, get_api_key
import orchestrator.tools as T

_BACKOFF_SECONDS = [2, 4, 8, 16, 32]

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


def _strip_think(text: str) -> str:
    # Replace with a space so adjacent words around a mid-sentence block aren't concatenated.
    return _THINK_RE.sub(" ", text).strip()


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


def run(
    persona: dict,
    messages: list[dict],
    cwd: str | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    verbose: bool = True,
    api_key: str | None = None,
    exclude_tools: set[str] | None = None,
) -> tuple[str, list[dict]]:
    provider = persona.get("provider", "litellm")
    client = _client(provider, api_key)
    history = [{"role": "system", "content": persona["system"]}] + messages

    blocked = _ORCHESTRATOR_ONLY | (exclude_tools or set())
    agent_tools = [t for t in T.DEFS if t["function"]["name"] not in blocked]

    for round_num in range(max_tool_rounds):
        kwargs = dict(
            model=persona["model"],
            messages=history,
            tools=agent_tools,
            tool_choice="auto",
        )
        if "max_tokens" in persona:
            kwargs["max_tokens"] = persona["max_tokens"]
        for attempt in range(len(_BACKOFF_SECONDS) + 1):
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
                        print(
                            f"  [{persona['name']}] rate limit (attempt {attempt + 1}/5):"
                            f" reset at {reset_time.strftime('%Y-%m-%d %H:%M:%S UTC')};"
                            f" sleeping {sleep_secs:.1f}s"
                        )
                    time.sleep(sleep_secs)
                else:
                    next_wait = _BACKOFF_SECONDS[attempt]
                    if verbose:
                        print(
                            f"  [{persona['name']}] rate limit (attempt {attempt + 1}/5):"
                            f" unknown reset time (using backoff); retrying in {next_wait}s"
                        )
                    time.sleep(next_wait)
            except (openai.InternalServerError, openai.APIConnectionError) as e:
                if attempt >= len(_BACKOFF_SECONDS):
                    raise
                next_wait = _BACKOFF_SECONDS[attempt]
                if verbose:
                    print(
                        f"  [{persona['name']}] transient error (attempt {attempt + 1}/5):"
                        f" {e}; retrying in {next_wait}s"
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

        # Append assistant turn (convert to dict safely)
        assistant_entry = {"role": "assistant", "content": content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        history.append(assistant_entry)

        if not msg.tool_calls:
            return content, history

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                # Surface malformed JSON back to the model so it can retry
                # rather than crashing the entire run (see #133).
                error_msg = f"ERROR: could not parse tool arguments as JSON: {exc}"
                if verbose:
                    print(f"  [{persona['name']}] {error_msg}")
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": error_msg,
                })
                continue
            if verbose:
                print(f"  [{persona['name']}] tool: {name}({list(args.keys())})")
            result = T.execute(name, args, cwd=cwd)
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    raise MaxToolRoundsError(persona["name"], max_tool_rounds, history)
