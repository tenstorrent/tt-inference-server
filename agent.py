"""Single-agent loop: run a persona until it produces a text response."""

import json
from openai import OpenAI
from config import LITELLM_BASE_URL, get_api_key
import tools as T

def _client() -> OpenAI:
    return OpenAI(base_url=LITELLM_BASE_URL, api_key=get_api_key())

def run(
    persona: dict,
    messages: list[dict],
    cwd: str | None = None,
    max_tool_rounds: int = 20,
    verbose: bool = True,
) -> tuple[str, list[dict]]:
    """
    Run a persona against a message history.
    Returns (final_text, updated_messages_including_system).
    """
    client = _client()
    history = [{"role": "system", "content": persona["system"]}] + messages

    for round_num in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=persona["model"],
            messages=history,
            tools=T.DEFS,
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
            # Done — return the text response
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

    return f"ERROR: hit max_tool_rounds ({max_tool_rounds}) without finishing", history
