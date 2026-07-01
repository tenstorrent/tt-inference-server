"""Single-agent loop: run a persona until it produces a text response."""

import json
from openai import OpenAI
from orchestrator.config import LITELLM_BASE_URL, get_api_key
import orchestrator.tools as T

# Default hard cap on tool-call iterations.  This is a safety rail, not a
# cost-control mechanism — cost is better managed at the token/dollar level.
# Callers can lower it for simple tasks or raise it for complex ones by
# passing max_tool_rounds explicitly to run().
DEFAULT_MAX_TOOL_ROUNDS = 100

# Tools the orchestrator calls directly; agents must never invoke them.
_ORCHESTRATOR_ONLY = frozenset({"create_pr"})


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


def _client(api_key: str | None = None) -> OpenAI:
    return OpenAI(base_url=LITELLM_BASE_URL, api_key=get_api_key(api_key))

def run(
    persona: dict,
    messages: list[dict],
    cwd: str | None = None,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    verbose: bool = True,
    api_key: str | None = None,
    exclude_tools: set[str] | None = None,
) -> tuple[str, list[dict]]:
    client = _client(api_key)
    history = [{"role": "system", "content": persona["system"]}] + messages

    blocked = _ORCHESTRATOR_ONLY | (exclude_tools or set())
    agent_tools = [t for t in T.DEFS if t["function"]["name"] not in blocked]

    for round_num in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=persona["model"],
            messages=history,
            tools=agent_tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        # Append assistant turn (convert to dict safely)
        assistant_entry = {"role": "assistant", "content": msg.content or ""}
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
            return msg.content or "", history

        # Execute each tool call and append results
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            if verbose:
                print(f"  [{persona['name']}] tool: {name}({list(args.keys())})")
            result = T.execute(name, args, cwd=cwd)
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    raise MaxToolRoundsError(persona["name"], max_tool_rounds, history)
