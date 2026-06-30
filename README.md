# multi-agent

Debate-and-consensus multi-agent orchestrator for automated code changes.

## How it works

1. **Implementer** explores the repo and makes the requested change
2. **Reviewers** (security, correctness) audit the diff in parallel
3. If any reviewer objects -> **debate round**: implementer addresses concerns, reviewers re-evaluate
4. Repeat up to `max_debate_rounds` (default 3)
5. If consensus -> PR opened.  If not -> exits non-zero with objection summary.

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

Pass an API key explicitly (falls back to env-var / key file when omitted):

```bash
python run.py --api-key sk-my-key /path/to/repo "add rate limiting to the /login endpoint"
```

**Security note:** `--api-key` exposes the value in process listings (`ps aux`)
and shell history. Prefer `TT_CHAT_API_KEY` or the key file for CI / non-interactive use.

## Adding personas

Edit `orchestrator/personas.py` — each persona is a dict with `name`, `model`, and `system`.
To use a different model for one persona (e.g. Kimi when it lands):

```python
SECURITY_REVIEWER = {
    "name": "security_reviewer",
    "model": "kimi/kimi-k2",   # swap here
    "system": "...",
}
```

## Repository layout

```
run.py                        # CLI entry point
requirements.txt              # Python dependencies
orchestrator/                 # Orchestration package
    __init__.py               #   exports orchestrate() as top-level public API
    config.py                 #   LiteLLM proxy URL, model defaults, key loading
    tools.py                  #   bash_exec, read_file, write_file, git_*, create_pr
    personas.py               #   Persona definitions (system prompts + model)
    agent.py                  #   Single-agent ReAct loop (handles tool calls)
    orchestrator.py           #   Debate/consensus loop + PR creation
project/                      # Future project code (empty placeholder)
```
