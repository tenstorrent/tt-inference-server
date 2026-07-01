"""
Model compatibility test harness.

Usage:
    python3 orchestrator/test_model_compat.py --model anthropic/claude-sonnet-4-6 --provider litellm
    python3 orchestrator/test_model_compat.py --list-models
"""

import argparse
import json
import pathlib
import re
import sys
import textwrap

# When invoked as `python3 orchestrator/test_model_compat.py`, Python prepends
# the orchestrator/ directory to sys.path.  That causes `import orchestrator`
# to find orchestrator.py (a file) inside that directory instead of the
# orchestrator/ package at the project root.  We strip any path entry that is
# the same directory as this script itself, then insert the project root, so
# that `import orchestrator.config` resolves correctly from any working directory.
_here = pathlib.Path(__file__).resolve().parent
_root = _here.parent
sys.path = [p for p in sys.path if pathlib.Path(p).resolve() != _here]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from openai import OpenAI

from orchestrator.config import (
    DEFAULT_MODEL,
    FAST_MODEL,
    PROVIDER_REGISTRY,
    get_api_key,
)
import orchestrator.tools as T


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(model: str, provider: str) -> OpenAI:
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider {provider!r}. Valid providers: {sorted(PROVIDER_REGISTRY)}"
        )
    entry = PROVIDER_REGISTRY[provider]
    key = get_api_key() if provider == "litellm" else entry["get_key"]()
    return OpenAI(base_url=entry["base_url"], api_key=key)


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return passed


# Markdown code-fence pattern that some models (e.g. DeepSeek-R1) emit instead of raw JSON.
_MARKDOWN_JSON_RE = re.compile(r"^\s*```", re.MULTILINE)


def _is_markdown_wrapped(text: str) -> bool:
    return bool(_MARKDOWN_JSON_RE.search(text))


# ---------------------------------------------------------------------------
# Check 1: tool-call compliance
# ---------------------------------------------------------------------------

def check_tool_call(client: OpenAI, model: str) -> bool:
    print("\nCheck 1: tool-call compliance")

    bash_exec_def = next(d for d in T.DEFS if d["function"]["name"] == "bash_exec")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the available tools to answer requests.",
        },
        {"role": "user", "content": "Run the command: echo hello"},
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[bash_exec_def],
        tool_choice="auto",
    )

    msg = response.choices[0].message
    if not msg.tool_calls:
        return _check("tool call issued", False, "model returned text instead of a tool call")

    _check("tool call issued", True)

    all_passed = True
    for tc in msg.tool_calls:
        name_ok = _check(
            "tool name is exact registered name",
            tc.function.name == "bash_exec",
            f"got {tc.function.name!r}",
        )

        # Detect markdown-fence wrapping before attempting JSON parse; some models
        # emit ```json\n{}\n``` instead of a bare JSON object.
        markdown_ok = _check(
            "arguments not markdown-wrapped",
            not _is_markdown_wrapped(tc.function.arguments),
            f"arguments start with: {tc.function.arguments[:80]!r}",
        )

        try:
            json.loads(tc.function.arguments)
            json_ok = _check("arguments are valid JSON", True)
        except json.JSONDecodeError as exc:
            json_ok = _check("arguments are valid JSON", False, str(exc))

        all_passed = all_passed and name_ok and markdown_ok and json_ok

    return all_passed


# ---------------------------------------------------------------------------
# Check 2: reviewer verdict format
# ---------------------------------------------------------------------------

def check_reviewer_verdict(client: OpenAI, model: str) -> bool:
    print("\nCheck 2: reviewer verdict format")

    system = textwrap.dedent("""\
        You are a senior engineer auditing a code change for correctness.

        Review the diff provided by the user and end your response with exactly one of:
          APPROVED
          OBJECTION: <concise list of specific concerns>
    """)

    # Minimal synthetic diff; content is irrelevant, we only care about the verdict.
    diff = textwrap.dedent("""\
        diff --git a/foo.py b/foo.py
        index 0000000..1111111 100644
        --- a/foo.py
        +++ b/foo.py
        @@ -1,3 +1,4 @@
         def greet(name):
        -    return "hello"
        +    return f"hello {name}"
        +
    """)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Please review this diff:\n\n{diff}"},
    ]

    response = client.chat.completions.create(model=model, messages=messages)

    text = response.choices[0].message.content or ""
    verdict_present = bool(re.search(r"\bAPPROVED\b|\bOBJECTION\b", text, re.IGNORECASE))

    return _check(
        "response contains APPROVED or OBJECTION",
        verdict_present,
        f"tail of response: {text[-200:].strip()!r}" if not verdict_present else "",
    )


# ---------------------------------------------------------------------------
# --list-models
# ---------------------------------------------------------------------------

def list_models() -> None:
    print("Models defined in orchestrator/config.py:")
    print(f"  DEFAULT_MODEL = {DEFAULT_MODEL!r}")
    print(f"  FAST_MODEL    = {FAST_MODEL!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test a model's compatibility with the orchestrator tool protocol."
    )
    parser.add_argument("--model", help="Model identifier, e.g. anthropic/claude-sonnet-4-6")
    parser.add_argument(
        "--provider",
        default="litellm",
        help="Provider name (default: litellm)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print all models defined in config.py and exit",
    )
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return 0

    if not args.model:
        parser.error("--model is required unless --list-models is specified")

    print(f"Testing model: {args.model!r}  provider: {args.provider!r}")

    try:
        client = _make_client(args.model, args.provider)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    results = [
        check_tool_call(client, args.model),
        check_reviewer_verdict(client, args.model),
    ]

    print("\n--- Summary ---")
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
