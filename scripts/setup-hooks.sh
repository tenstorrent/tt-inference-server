#!/bin/sh

# Setup git hooks for this repository.
# Run once after cloning: ./scripts/setup-hooks.sh

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
if [ -z "$REPO_ROOT" ]; then
    echo "❌ Not inside a git repository"
    exit 1
fi

cp "$REPO_ROOT/scripts/hooks/pre-commit" "$REPO_ROOT/.git/hooks/pre-commit"
chmod +x "$REPO_ROOT/.git/hooks/pre-commit"
echo "✅ pre-commit hook installed"
