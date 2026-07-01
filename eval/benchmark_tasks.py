"""Benchmark task definitions: 5 implementer tasks + 5 reviewer tasks."""


# ---------------------------------------------------------------------------
# check_fn helpers
# ---------------------------------------------------------------------------

def _check_core(content: str) -> bool:
    return "if not items" in content or "items is None" in content


def _check_paginator(content: str) -> bool:
    return "(page - 1)" in content or "page-1" in content


def _check_auth(content: str) -> bool:
    return "return True" in content


def _check_formatter(content: str) -> bool:
    return 'f"' in content or "f'" in content


def _check_cache(content: str) -> bool:
    lines = content.split("\n")
    return any(
        "time.time()" in line and ("-" in line or "<" in line)
        for line in lines
    )


# ---------------------------------------------------------------------------
# Implementer tasks
# ---------------------------------------------------------------------------

IMPLEMENTER_TASKS = [
    {
        "id": "core-guard",
        "label": "core.py guard clause",
        "prompt": (
            "In `core.py`, the `process(items)` function crashes when called with `None`.\n"
            "Add a guard clause at the top so that `process(None)` and `process([])` both\n"
            "return an empty list without raising an exception.\n"
            "Only change `core.py`. Run `python3 -m pytest tests/test_core.py -v` to verify."
        ),
        "fixture_file": "core.py",
        "check_fn": _check_core,
        "test_command": "python3 -m pytest tests/test_core.py -v",
    },
    {
        "id": "paginator-off-by-one",
        "label": "paginator.py off-by-one",
        "prompt": (
            "In `paginator.py`, the `paginate(items, page_size, page)` function has an\n"
            "off-by-one error. The function should be 1-indexed: page=1 returns\n"
            "items[0:page_size], page=2 returns items[page_size:2*page_size], etc.\n"
            "Fix the slice calculation.\n"
            "Only change `paginator.py`. Run `python3 -m pytest tests/test_paginator.py -v` to verify."
        ),
        "fixture_file": "paginator.py",
        "check_fn": _check_paginator,
        "test_command": "python3 -m pytest tests/test_paginator.py -v",
    },
    {
        "id": "auth-missing-return",
        "label": "auth.py missing return",
        "prompt": (
            "In `auth.py`, the `validate(token)` function is missing a return statement.\n"
            "When the token passes all validation checks the function should return `True`,\n"
            "but currently it falls off the end and returns `None` implicitly.\n"
            "Only change `auth.py`. Run `python3 -m pytest tests/test_auth.py -v` to verify."
        ),
        "fixture_file": "auth.py",
        "check_fn": _check_auth,
        "test_command": "python3 -m pytest tests/test_auth.py -v",
    },
    {
        "id": "formatter-concat",
        "label": "formatter.py concat",
        "prompt": (
            "In `formatter.py`, the `format_price(amount, currency)` function uses `+`\n"
            "to concatenate `currency` and `amount`. This crashes with TypeError when\n"
            "`amount` is a float. Fix it to produce a string like \"USD 9.99\"\n"
            "(currency, space, amount to 2 decimal places).\n"
            "Only change `formatter.py`. Run `python3 -m pytest tests/test_formatter.py -v` to verify."
        ),
        "fixture_file": "formatter.py",
        "check_fn": _check_formatter,
        "test_command": "python3 -m pytest tests/test_formatter.py -v",
    },
    {
        "id": "cache-ttl",
        "label": "cache.py ttl check",
        "prompt": (
            "In `cache.py`, the `get_or_set(key, fn, ttl)` function never checks whether\n"
            "the cached value has expired. It should compare the current time against the\n"
            "stored timestamp and call `fn()` again when `ttl` seconds have elapsed.\n"
            "Only change `cache.py`. Run `python3 -m pytest tests/test_cache.py -v` to verify."
        ),
        "fixture_file": "cache.py",
        "check_fn": _check_cache,
        "test_command": "python3 -m pytest tests/test_cache.py -v",
    },
]


# ---------------------------------------------------------------------------
# Reviewer tasks — synthetic diffs
# ---------------------------------------------------------------------------

_OFF_BY_ONE_DIFF = """\
diff --git a/api/pagination.py b/api/pagination.py
--- a/api/pagination.py
+++ b/api/pagination.py
@@ -10,7 +10,9 @@ class PagedResult:

 def get_page(items: list, page_size: int, page: int) -> list:
     \"\"\"Return a page of items. page is 1-indexed.\"\"\"
-    return items[:page_size]  # always returns first page
+    start = (page + 1) * page_size
+    end = start + page_size
+    return items[start:end]
"""

_MISSING_GUARD_DIFF = """\
diff --git a/services/processor.py b/services/processor.py
--- a/services/processor.py
+++ b/services/processor.py
@@ -1,3 +1,10 @@
+def sanitize_inputs(data: dict) -> dict:
+    \"\"\"Normalize incoming request data before processing.\"\"\"
+    cleaned = {}
+    for key, value in data.items():
+        cleaned[key] = value.strip()
+    return cleaned
+
+
 def process_request(request):
     return handle(request)
"""

_WRONG_OPERATOR_DIFF = """\
diff --git a/utils/formatting.py b/utils/formatting.py
--- a/utils/formatting.py
+++ b/utils/formatting.py
@@ -12,5 +12,5 @@ MAX_LABEL_LEN = 64

 def build_label(prefix: str, count: int) -> str:
     \"\"\"Format a label string like 'errors: 42'.\"\"\"
-    return f\"{prefix}: {count}\"
+    return prefix + \": \" + count
"""

_CLEAN_REFACTOR_DIFF = """\
diff --git a/utils/helpers.py b/utils/helpers.py
--- a/utils/helpers.py
+++ b/utils/helpers.py
@@ -5,8 +5,6 @@ import re

 def normalize_tag(tag: str) -> str:
     \"\"\"Lowercase and strip whitespace from a tag.\"\"\"
-    tag = tag.lower()
-    tag = tag.strip()
-    return tag
+    return tag.lower().strip()
"""

_CLEAN_VALIDATION_DIFF = """\
diff --git a/api/users.py b/api/users.py
--- a/api/users.py
+++ b/api/users.py
@@ -15,6 +15,9 @@ from typing import Optional

 def get_user(user_id: int) -> Optional[User]:
     \"\"\"Fetch a user by their numeric ID.\"\"\"
+    if user_id <= 0:
+        raise ValueError(f\"user_id must be positive, got {user_id}\")
     return db.session.query(User).filter(User.id == user_id).first()
"""

REVIEWER_TASKS = [
    {
        "id": "review-off-by-one",
        "label": "off-by-one diff",
        "diff": _OFF_BY_ONE_DIFF,
        "expected_verdict": "OBJECTION",
        "description": "start uses (page+1)*page_size instead of (page-1)*page_size — skips to wrong page",
    },
    {
        "id": "review-missing-guard",
        "label": "missing guard diff",
        "diff": _MISSING_GUARD_DIFF,
        "expected_verdict": "OBJECTION",
        "description": "value.strip() crashes with AttributeError when value is not a string",
    },
    {
        "id": "review-wrong-operator",
        "label": "wrong operator diff",
        "diff": _WRONG_OPERATOR_DIFF,
        "expected_verdict": "OBJECTION",
        "description": "prefix + \": \" + count raises TypeError — count is int, not str",
    },
    {
        "id": "review-clean-refactor",
        "label": "clean diff 1",
        "diff": _CLEAN_REFACTOR_DIFF,
        "expected_verdict": "APPROVED",
        "description": "functionally identical refactor — no behavior change",
    },
    {
        "id": "review-clean-validation",
        "label": "clean diff 2",
        "diff": _CLEAN_VALIDATION_DIFF,
        "expected_verdict": "APPROVED",
        "description": "valid input guard using ValueError and f-string — correct",
    },
]
