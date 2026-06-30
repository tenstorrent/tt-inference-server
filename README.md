# multi-agent

Debate-and-consensus multi-agent orchestrator for automated code changes.

## How it works

1. **Implementer** explores the repo and makes the requested change
2. **Reviewers** (security, correctness) audit the diff in parallel
3. If any reviewer objects → **debate round**: implementer addresses concerns, reviewers re-evaluate
4. Repeat up to `max_debate_rounds` (default 3)
5. If consensus → PR opened.  If not → exits non-zero with objection summary.

## Setup

```bash
pip install openai
```

The key is read from `/workspace/global/.litellm.key` or `$TT_CHAT_API_KEY`.

## Usage

```bash
cd multi-agent
python run.py /path/to/repo "add rate limiting to the /login endpoint"
```

## Adding personas

Edit `personas.py` — each persona is a dict with `name`, `model`, and `system`.
To use a different model for one persona (e.g. Kimi when it lands):

```python
SECURITY_REVIEWER = {
    "name": "security_reviewer",
    "model": "kimi/kimi-k2",   # swap here
    "system": "...",
}
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | LiteLLM proxy URL, model defaults, key loading |
| `tools.py` | bash_exec, read_file, write_file, git_*, create_pr |
| `personas.py` | Persona definitions (system prompts + model) |
| `agent.py` | Single-agent ReAct loop (handles tool calls) |
| `orchestrator.py` | Debate/consensus loop + PR creation |
| `run.py` | CLI entry point |
