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

# -- internal helper: shell=False gh runner -----------------------------------

# All five issue-management functions use _gh() instead of bash_exec().
# Building an argv list and passing shell=False means the kernel does a direct
# execve(); there is no shell interpreter to misparse quotes, semicolons, pipes,
# backticks, or any other metacharacter.  LLM-supplied strings (state, labels,
# issue numbers, comment bodies) are therefore passed verbatim as arguments with
# zero injection risk — no shlex.quote() needed and no quoting edge-cases to miss.

_ALLOWED_STATES = frozenset({"open", "closed", "all"})

def _gh(argv: list[str], cwd: str | None = None) -> str:
    """Run  gh <argv>  with shell=False and return stdout+stderr (≤ 8 k chars)."""
    r = subprocess.run(
        ["gh"] + argv,
        shell=False,          # ← no shell; metacharacters in argv are inert
        capture_output=True,
        text=True,
        timeout=120,
        cwd=cwd,
    )
    out = r.stdout + r.stderr
    return out[:8000] if len(out) > 8000 else out

# -- issue management tools ---------------------------------------------------

def list_issues(
    state: str = "open",
    labels: str = "",
    limit: int = 50,
    cwd: str | None = None,
) -> str:
    """List GitHub issues using the ``gh`` CLI.

    Args:
        state:  Issue state filter — must be one of "open", "closed", "all".
                Values outside this set are rejected before reaching the shell.
        labels: Comma-separated label names to filter by (empty = no filter).
                Each name is passed as a separate argv token; shell metacharacters
                in a label name are therefore inert.
        limit:  Maximum number of issues to return (1–200, default 50).
        cwd:    Working directory for the gh command.

    Returns:
        JSON array of issue objects as a string (truncated at 8 k chars).
    """
    # Validate state against the known-good set so we surface bad input early
    # rather than silently passing an attacker-controlled token to gh.
    if state not in _ALLOWED_STATES:
        return f"ERROR: invalid state {state!r}; must be one of {sorted(_ALLOWED_STATES)}"

    limit = max(1, min(int(limit), 200))

    argv = [
        "issue", "list",
        "--state", state,          # safe: validated above
        "--limit", str(limit),     # safe: already cast to int
        "--json", "number,title,labels,state,body,assignees,createdAt,updatedAt",
    ]

    if labels:
        for label in labels.split(","):
            label = label.strip()
            if label:
                # Each label is its own argv element — no quoting needed
                argv += ["--label", label]

    return _gh(argv, cwd=cwd)


def get_issue(number: int, cwd: str | None = None) -> str:
    """Fetch full details of a single GitHub issue.

    Args:
        number: The issue number.  Cast to ``int`` to prevent non-numeric injection.
        cwd:    Working directory for the gh command.

    Returns:
        JSON object for the issue (truncated at 8 k chars).
    """
    argv = [
        "issue", "view", str(int(number)),   # int() rejects non-numeric input
        "--json", "number,title,labels,state,body,comments,assignees,createdAt,updatedAt",
    ]
    return _gh(argv, cwd=cwd)


def comment_issue(number: int, body: str, cwd: str | None = None) -> str:
    """Post a comment on a GitHub issue.

    Args:
        number: The issue number.
        body:   Markdown text of the comment.  Written to a temp file so that
                even a body containing shell metacharacters is safe.
        cwd:    Working directory for the gh command.

    Returns:
        Output of the gh command.
    """
    import tempfile, pathlib
    # Write body to a temp file.  With shell=False the --body-file path is
    # passed as a plain argv token; the body content itself never touches a shell.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(body)
        tmp_path = f.name
    try:
        argv = ["issue", "comment", str(int(number)), "--body-file", tmp_path]
        return _gh(argv, cwd=cwd)
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)


def label_issue(number: int, labels: str, cwd: str | None = None) -> str:
    """Apply one or more labels to a GitHub issue (additive — existing labels are kept).

    Args:
        number: The issue number.
        labels: Comma-separated label names to add.  Each name becomes a
                separate argv element; shell metacharacters are inert.
        cwd:    Working directory for the gh command.

    Returns:
        Output of the gh command.
    """
    label_list = [l.strip() for l in labels.split(",") if l.strip()]
    if not label_list:
        return "ERROR: no labels provided"

    argv = ["issue", "edit", str(int(number))]
    for label in label_list:
        # "--add-label" and the label value are separate argv tokens
        argv += ["--add-label", label]

    return _gh(argv, cwd=cwd)


def close_issue(number: int, reason: str = "", cwd: str | None = None) -> str:
    """Close a GitHub issue, optionally leaving a closing comment first.

    Args:
        number: The issue number.
        reason: Optional comment text to post before closing.
        cwd:    Working directory for the gh command.

    Returns:
        Output of the gh command(s).
    """
    results = []
    if reason:
        results.append(comment_issue(number, reason, cwd=cwd))
    results.append(_gh(["issue", "close", str(int(number))], cwd=cwd))
    return "\n".join(results)

# -- dispatch -----------------------------------------------------------------

IMPL = {
    "bash_exec":     lambda args, cwd: bash_exec(args["command"], cwd),
    "read_file":     lambda args, cwd: read_file(args["path"]),
    "write_file":    lambda args, cwd: write_file(args["path"], args["content"]),
    "git_status":    lambda args, cwd: git_status(cwd),
    "git_diff":      lambda args, cwd: git_diff(cwd),
    "create_pr":     lambda args, cwd: create_pr(args["title"], args["body"], args["branch"], cwd),
    "list_issues":   lambda args, cwd: list_issues(
                         args.get("state", "open"),
                         args.get("labels", ""),
                         args.get("limit", 50),
                         cwd,
                     ),
    "get_issue":     lambda args, cwd: get_issue(args["number"], cwd),
    "comment_issue": lambda args, cwd: comment_issue(args["number"], args["body"], cwd),
    "label_issue":   lambda args, cwd: label_issue(args["number"], args["labels"], cwd),
    "close_issue":   lambda args, cwd: close_issue(args["number"], args.get("reason", ""), cwd),
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
    {
        "type": "function",
        "function": {
            "name": "list_issues",
            "description": (
                "List GitHub issues via the gh CLI. "
                "Returns a JSON array with fields: number, title, labels, state, body, "
                "assignees, createdAt, updatedAt."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Filter by issue state (default: open).",
                    },
                    "labels": {
                        "type": "string",
                        "description": "Comma-separated label names to filter by (optional).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum issues to return, 1-200 (default: 50).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_issue",
            "description": (
                "Fetch full details of a single GitHub issue including its body and comments."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue number.",
                    },
                },
                "required": ["number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "comment_issue",
            "description": "Post a markdown comment on a GitHub issue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue number.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Markdown text of the comment.",
                    },
                },
                "required": ["number", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "label_issue",
            "description": (
                "Add one or more labels to a GitHub issue "
                "(additive; existing labels are kept)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue number.",
                    },
                    "labels": {
                        "type": "string",
                        "description": "Comma-separated label names to add.",
                    },
                },
                "required": ["number", "labels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_issue",
            "description": (
                "Close a GitHub issue, optionally posting a closing comment first "
                "(e.g. 'duplicate of #42' or 'out of scope')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "description": "The issue number.",
                    },
                    "reason": {
                        "type": "string",
                        "description": (
                            "Optional comment to post before closing, e.g. "
                            "'Duplicate of #12' or 'Out of scope for v1'."
                        ),
                    },
                },
                "required": ["number"],
            },
        },
    },
]
