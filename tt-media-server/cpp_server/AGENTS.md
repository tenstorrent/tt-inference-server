# AGENTS.md

Instructions for agentic coding tools (Claude Code, Cursor, Codex, etc.) working in `cpp_server`.

## Requirements and Issues

Requirements live in GitHub Issues at
[tenstorrent/tt-inference-server](https://github.com/tenstorrent/tt-inference-server/issues).

Issues relevant to this component are labeled with **both** `cpp-server` and
`Inference technologies`. Use `gh` to read them:

```bash
# List open issues for cpp_server
gh issue list --repo tenstorrent/tt-inference-server \
  --label cpp-server --label "Inference technologies"

# View a specific issue
gh issue view <number> --repo tenstorrent/tt-inference-server
```

Issues here are typically terse. If an issue doesn't give you enough context
to implement it (no acceptance criteria, ambiguous scope, unclear definition
of done), **ask the user before implementing** — don't infer scope from the
title alone.
