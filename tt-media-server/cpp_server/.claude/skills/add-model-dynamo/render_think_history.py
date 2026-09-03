#!/usr/bin/env python3
"""Derive thinkStartInHistory / thinkEndInHistory for a model's StaticTokenizerInfo.

Renders each model's chat template for a two-turn conversation and reports which
think delimiters survive in the HISTORY (the part before the new user turn).
A delimiter that survives is re-supplied by the next prompt; one that does not
is a KV row only accumulatedThinkTokens can account for.

    python3 render_think_history.py                    # every fetched model
    python3 render_think_history.py zai-org/GLM-5.1    # one model

Requires jinja2 (no model weights, no tokenizer load — templates only).
"""
import datetime
import glob
import json
import os
import sys

from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

TOKENIZERS = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../tokenizers")
)
MARKER_HINTS = ("think", "channel")
TURN2 = "SECOND_USER_TURN"


def load_template(model_dir):
    path = os.path.join(model_dir, "chat_template.jinja")
    if os.path.exists(path):
        return open(path).read()
    cfg = json.load(open(os.path.join(model_dir, "tokenizer_config.json")))
    tpl = cfg.get("chat_template")
    return tpl[0]["template"] if isinstance(tpl, list) else tpl


def bos_token(model_dir):
    tok = json.load(open(os.path.join(model_dir, "tokenizer_config.json"))).get("bos_token")
    return (tok.get("content") if isinstance(tok, dict) else tok) or ""


def special_tokens(model_dir):
    """All special tokens as {content: id}, from tokenizer.json when present and
    from tokenizer_config.json's added_tokens_decoder otherwise (Kimi ships a
    tiktoken.model instead of a tokenizer.json)."""
    tokens = {}
    path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(path):
        for t in json.load(open(path)).get("added_tokens", []):
            tokens[t["content"]] = t["id"]
    cfg_path = os.path.join(model_dir, "tokenizer_config.json")
    if os.path.exists(cfg_path):
        decoder = json.load(open(cfg_path)).get("added_tokens_decoder", {})
        for tok_id, entry in decoder.items():
            content = entry.get("content") if isinstance(entry, dict) else entry
            if content:
                tokens.setdefault(content, int(tok_id))
    return tokens


def marker_candidates(model_dir):
    """Special tokens that look like think delimiters, as {content: id}."""
    return {
        content: tok_id
        for content, tok_id in special_tokens(model_dir).items()
        if any(h in content.lower() for h in MARKER_HINTS)
    }


def render(tpl, messages, bos, **kwargs):
    # loopcontrols: some templates (Kimi) use {% break %} inside their history
    # loop, which plain jinja2 rejects.
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"])
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(TemplateError(m))
    env.globals["strftime_now"] = lambda f: datetime.datetime.now().strftime(f)
    env.filters["tojson"] = lambda x, **kw: json.dumps(x)
    return env.from_string(tpl).render(
        messages=messages, add_generation_prompt=True, bos_token=bos,
        eos_token="", tools=None, **kwargs)


def conversation(echo_reasoning, with_assistant=True):
    turns = [{"role": "user", "content": "FIRST_USER_TURN"}]
    if with_assistant:
        assistant = {"role": "assistant", "content": "FIRST_ANSWER"}
        if echo_reasoning:
            assistant["reasoning_content"] = "FIRST_REASONING"
            assistant["reasoning"] = "FIRST_REASONING"
        turns.append(assistant)
    turns.append({"role": "user", "content": TURN2})
    return turns


def history_of(prompt):
    """Everything before the new user turn — i.e. what later turns re-render."""
    idx = prompt.find(TURN2)
    return prompt if idx < 0 else prompt[:idx]


def report(model, model_dir, thinking_kwargs):
    tpl = load_template(model_dir)
    if not tpl:
        print(f"{model}: no chat template fetched — cannot verify\n")
        return
    bos = bos_token(model_dir)
    markers = marker_candidates(model_dir)
    print(f"{model}")
    if not markers:
        print("  no think-like special tokens found "
              "(non-reasoning model → leave both flags at their default)\n")
        return
    for echo in (False, True):
        hist = history_of(render(tpl, conversation(echo), bos, **thinking_kwargs))
        # A template may DOCUMENT its delimiters in the system prompt (MiniMax
        # does), so a bare substring search over the history lies. Diff against
        # the same conversation minus the assistant turn: only the delta was
        # contributed by rendering a past think block.
        baseline = history_of(
            render(tpl, conversation(echo, with_assistant=False), bos,
                   **thinking_kwargs))
        label = ("reasoning echoed back" if echo
                 else "reasoning NOT echoed (use this one)")
        print(f"  [{label}]")
        print(f"    assistant turn renders as: {hist[len(os.path.commonprefix([hist, baseline])):]!r}")
        for content, tok_id in sorted(markers.items(), key=lambda kv: kv[1]):
            kept = hist.count(content) > baseline.count(content)
            print(f"    {content:<14} id={tok_id:<7} "
                  f"{'kept in history  -> InHistory=true' if kept else 'dropped          -> InHistory=false'}")
    print()


def main():
    wanted = sys.argv[1:]
    dirs = sorted(glob.glob(os.path.join(TOKENIZERS, "*", "*")))
    if wanted:
        dirs = [d for d in dirs if "/".join(d.split("/")[-2:]) in wanted]
        if not dirs:
            sys.exit(f"no fetched tokenizer matched {wanted} under {TOKENIZERS}")
    for d in dirs:
        model = "/".join(d.split("/")[-2:])
        try:
            report(model, d, {"enable_thinking": True})
        except Exception as exc:  # template bugs / unsupported jinja extensions
            print(f"{model}: RENDER FAILED: {type(exc).__name__}: {exc}")
            print("  fall back to reading the template's history branch by hand\n")


if __name__ == "__main__":
    main()
