import subprocess, os, json

# -- implementations ----------------------------------------------------------

def bash_exec(command: str, cwd: str | None = None) -> str:
    r = subprocess.run(command, shell=True, capture_output=True, text=True,
                       timeout=120, cwd=cwd)
    out = r.stdout + r.stderr
    return out[:8000] if len(out) > 8000 else out  # guard against huge output

def read_file(path: str) -> str:
    try:
        with open(path) as f:
            content = f.read()
        return content[:8000] if len(content) > 8000 else content
    except Exception as e:
        return f"ERROR: {e}"

def write_file(path: str, content: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"

def git_status(cwd: str | None = None) -> str:
    return bash_exec("git status --short && git log --oneline -5", cwd=cwd)

def git_diff(cwd: str | None = None) -> str:
    return bash_exec("git diff HEAD", cwd=cwd)

def create_pr(title: str, body: str, branch: str, cwd: str | None = None) -> str:
    cmds = [
        f"git checkout -b {branch}",
        "git add -A",
        f'git commit -m "{title}"',
        f"git push -u origin {branch}",
        f'gh pr create --title "{title}" --body "{body}"',
    ]
    results = []
    for cmd in cmds:
        out = bash_exec(cmd, cwd=cwd)
        results.append(f"$ {cmd}\n{out}")
        if "error" in out.lower() and "nothing to commit" not in out.lower():
            results.append("(stopping on error)")
            break
    return "\n".join(results)

# -- dispatch -----------------------------------------------------------------

IMPL = {
    "bash_exec":   lambda args, cwd: bash_exec(args["command"], cwd),
    "read_file":   lambda args, cwd: read_file(args["path"]),
    "write_file":  lambda args, cwd: write_file(args["path"], args["content"]),
    "git_status":  lambda args, cwd: git_status(cwd),
    "git_diff":    lambda args, cwd: git_diff(cwd),
    "create_pr":   lambda args, cwd: create_pr(args["title"], args["body"], args["branch"], cwd),
}

def execute(name: str, arguments: dict, cwd: str | None = None) -> str:
    fn = IMPL.get(name)
    if not fn:
        return f"ERROR: unknown tool {name}"
    try:
        return fn(arguments, cwd)
    except Exception as e:
        return f"ERROR: {e}"

# -- OpenAI-format schema -----------------------------------------------------

DEFS = [
    {
        "type": "function",
        "function": {
            "name": "bash_exec",
            "description": "Run a bash command. Returns stdout+stderr (truncated at 8k chars).",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file. Returns content (truncated at 8k chars).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent dirs as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show git status and recent commits.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show the current git diff against HEAD.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_pr",
            "description": "Commit all staged changes, push a new branch, and open a GitHub PR.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title":  {"type": "string", "description": "PR title"},
                    "body":   {"type": "string", "description": "PR description (markdown)"},
                    "branch": {"type": "string", "description": "New branch name, e.g. ai/fix-login-validation"},
                },
                "required": ["title", "body", "branch"],
            },
        },
    },
]
