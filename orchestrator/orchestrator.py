"""
Debate-and-consensus orchestrator.

Flow (PR mode):
  1. Implementer makes changes
  2. All reviewers audit in parallel (sequentially for now, easy to thread later)
  3. If any objections -> debate round: implementer rebuts/revises, reviewers re-evaluate
  4. Repeat up to max_debate_rounds
  5. If consensus -> create PR.  If not -> dump state and exit non-zero.

Flow (groom mode):
  1. Groomer reads open issues (passed as context) and applies labels / comments / closures
  2. All groom-reviewers audit the proposed actions in parallel
  3. If any objections -> debate round: groomer revises, reviewers re-evaluate
  4. Repeat up to max_debate_rounds
  5. If consensus -> report success.  If not -> dump state and exit non-zero.
"""

import copy, re, textwrap
from orchestrator.personas import (
    IMPLEMENTER, REVIEWERS, ACCEPTANCE_REVIEWER,
    GROOMER, GROOM_REVIEWERS,
)
import orchestrator.agent as A
from orchestrator.agent import MaxToolRoundsError, DEFAULT_MAX_TOOL_ROUNDS, REASONING_INJECTION_PREFIX

# close_issue is only valid for the groomer (closing duplicates). The
# implementer must not close issues — that happens via Closes #N at merge time.
# create_issue, add_sub_issue, remove_label are groomer-only: the implementer creates code, not issues.
_IMPLEMENTER_EXCLUDED_TOOLS = {"close_issue", "create_issue", "add_sub_issue", "remove_label"}


def _apply_persona_override(persona: dict, override: dict) -> dict:
    """Return a deep copy of persona with override fields merged in."""
    result = copy.deepcopy(persona)
    result.update(override)
    return result


def _parse_issue_number(task: str) -> int | None:
    # Mirrors run.parse_issue_number; kept local to avoid cross-package imports.
    m = re.search(r"#(\d+)", task)
    return int(m.group(1)) if m else None

def _extract_verdict(text: str) -> tuple[bool, str]:
    """Returns (approved, objection_text).

    Scans from the bottom up so the latest verdict wins — a reviewer who says
    "my previous FINDING is resolved — APPROVED" correctly counts as approved.
    FINDING: is treated as a third verdict branch alongside APPROVED/OBJECTION:
    if the bottom-up scan hits FINDING: before it hits APPROVED, it is a reject.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip().upper()
        if stripped == "APPROVED" or stripped.startswith("APPROVED"):
            return True, ""
        if stripped.startswith("OBJECTION"):
            return False, line.strip()
        # Severity doesn't soften a finding — it belongs in the text for
        # post-merge triage, not as justification to vote APPROVED.
        if stripped.startswith("FINDING:"):
            return False, line.strip()
    # No explicit verdict line found - check if APPROVED appears anywhere at end
    if "APPROVED" in text[-300:].upper():
        return True, ""
    return False, text[-500:]


def _extract_last_reasoning(history: list[dict]) -> str | None:
    # Scan backwards for the most-recent injected reasoning system message.
    for msg in reversed(history):
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if content.startswith(REASONING_INJECTION_PREFIX):
                # Strip the wrapper lines; return just the text inside <reasoning>.
                # Greedy .* matches up to the LAST </reasoning>, so embedded tags in
                # the content (which have no outer newline+tag to anchor against) are
                # captured intact rather than truncating at the first occurrence.
                m = re.search(r"<reasoning>\n(.*)\n</reasoning>", content, re.DOTALL)
                return m.group(1) if m else None
    return None


def _strip_reasoning_messages(history: list[dict]) -> list[dict]:
    # Reasoning injection system messages are implementer-internal; strip them
    # so reviewer A.run() calls never see mid-conversation system roles.
    return [
        m for m in history
        if not (m.get("role") == "system" and m.get("content", "").startswith(REASONING_INJECTION_PREFIX))
    ]


def _build_reviewer_messages(
    reviewer: dict,
    shared_history: list[dict],
    task: str,
    prompt: str,
) -> list[dict]:
    """Build the message list for a reviewer call.

    The acceptance reviewer receives the original task prompt prepended as a
    user message so it can check every stated and implied criterion.  Other
    reviewers receive the shared history unchanged.
    """
    if reviewer["name"] == ACCEPTANCE_REVIEWER["name"]:
        task_context = textwrap.dedent(f"""
            The following is the original task prompt that the implementer was asked to complete.
            Use it to verify that every stated and implied acceptance criterion has been met.

            --- ORIGINAL TASK ---
            {task}
            --- END ORIGINAL TASK ---
        """).strip()
        return (
            [{"role": "user", "content": task_context}]
            + shared_history
            + [{"role": "user", "content": prompt}]
        )
    return shared_history + [{"role": "user", "content": prompt}]


def orchestrate(
    task: str,
    repo_path: str,
    max_debate_rounds: int = 3,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    verbose: bool = True,
    api_key: str | None = None,
    implementer_override: dict | None = None,
    reviewer_override: dict | None = None,
) -> bool:
    """Returns True on success (PR opened), False on failure.

    Args:
        task:                   Natural-language description of the work to do.
        repo_path:              Absolute path to the target git repository.
        max_debate_rounds:      Maximum implementer <-> reviewer debate iterations.
        max_tool_rounds:        Hard cap on tool-call iterations per agent call.
        verbose:                Stream progress to stdout.
        api_key:                Optional LiteLLM API key.
        implementer_override:   Dict of fields to merge into the IMPLEMENTER persona
                                (e.g. {"model": "...", "provider": "..."}).
        reviewer_override:      Dict of fields to merge into every reviewer persona.
    """

    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    # Apply CLI overrides on top of persona defaults without mutating the originals.
    implementer = _apply_persona_override(IMPLEMENTER, implementer_override or {})
    reviewers = [
        _apply_persona_override(r, reviewer_override or {}) for r in REVIEWERS
    ]

    # -- Phase 1: Implementation ----------------------------------------------
    log("\n=== IMPLEMENTER ===")
    try:
        impl_text, impl_history = A.run(
            implementer,
            [{"role": "user", "content": task}],
            cwd=repo_path,
            max_tool_rounds=max_tool_rounds,
            verbose=verbose,
            api_key=api_key,
            exclude_tools=_IMPLEMENTER_EXCLUDED_TOOLS,
            inject_reasoning=True,
        )
    except MaxToolRoundsError as exc:
        log(f"\n=== IMPLEMENTER ABORTED: {exc} ===")
        log("Aborting run — implementer hit tool-round cap; work is incomplete.")
        return False
    log(f"[implementer] {impl_text[:300]}...")

    # Strip the system prompt and any reasoning injection messages; the latter
    # are implementer-internal and must not appear in reviewer context.
    shared_history = _strip_reasoning_messages(impl_history[1:])

    # -- Phase 2 + 3: Review / debate loop ------------------------------------
    for debate_round in range(max_debate_rounds + 1):
        if debate_round == 0:
            log("\n=== INITIAL REVIEW ===")
        else:
            log(f"\n=== DEBATE ROUND {debate_round} ===")

        verdicts: dict[str, tuple[bool, str]] = {}
        reviewer_histories: dict[str, list[dict]] = {}

        for reviewer in reviewers:
            log(f"\n-- {reviewer['name']} --")
            prompt = (
                "Please review the implementation and give your verdict."
                if debate_round == 0
                else "The implementer has revised. Please re-evaluate your previous objections."
            )
            messages = _build_reviewer_messages(reviewer, shared_history, task, prompt)
            review_text, rev_history = A.run(
                reviewer,
                messages,
                cwd=repo_path,
                max_tool_rounds=max_tool_rounds,
                verbose=verbose,
                api_key=api_key,
                exclude_tools=_IMPLEMENTER_EXCLUDED_TOOLS,
            )
            log(f"[{reviewer['name']}] {review_text[:300]}")
            approved, objection = _extract_verdict(review_text)
            verdicts[reviewer["name"]] = (approved, objection)
            reviewer_histories[reviewer["name"]] = rev_history

        objectors = {name: obj for name, (ok, obj) in verdicts.items() if not ok}

        if not objectors:
            log("\n=== CONSENSUS REACHED ===")
            break

        if debate_round == max_debate_rounds:
            log("\n=== FAILED TO REACH CONSENSUS ===")
            for name, obj in objectors.items():
                log(f"  {name}: {obj}")
            return False

        # Build debate context: all reviewer verdicts -> implementer responds
        log(f"\n-- implementer rebuttal (round {debate_round + 1}) --")
        objection_summary = "\n".join(
            f"- {name}: {obj}" for name, obj in objectors.items()
        )
        rebuttal_prompt = textwrap.dedent(f"""
            The following reviewers have objections:
            {objection_summary}

            Address each concern by revising the implementation or explaining why no change is needed.
            When done, end with: IMPLEMENTATION_COMPLETE
        """).strip()

        # Carry any reasoning the implementer produced into the rebuttal turn.
        last_reasoning = _extract_last_reasoning(impl_history)

        try:
            impl_text, impl_history = A.run(
                implementer,
                shared_history + [{"role": "user", "content": rebuttal_prompt}],
                cwd=repo_path,
                max_tool_rounds=max_tool_rounds,
                verbose=verbose,
                api_key=api_key,
                exclude_tools=_IMPLEMENTER_EXCLUDED_TOOLS,
                inject_reasoning=True,
                prior_reasoning=last_reasoning,
            )
        except MaxToolRoundsError as exc:
            log(f"\n=== IMPLEMENTER ABORTED (rebuttal round {debate_round + 1}): {exc} ===")
            log("Aborting run — implementer hit tool-round cap during rebuttal; work is incomplete.")
            return False
        log(f"[implementer] {impl_text[:300]}...")
        shared_history = _strip_reasoning_messages(impl_history[1:])

    # -- Phase 4: Open PR -----------------------------------------------------
    log("\n=== OPENING PR ===")
    review_summary = "\n".join(
        f"- **{name}**: {'approved' if ok else obj}"
        for name, (ok, obj) in verdicts.items()
    )
    # Sentinels are column-0 tokens that break textwrap.dedent's common-indent
    # calculation, and they have no business appearing in a public PR body.
    _SENTINELS = ("IMPLEMENTATION_COMPLETE",)
    clean_impl = impl_text
    for _s in _SENTINELS:
        clean_impl = clean_impl.replace(_s, "").strip()

    issue_number = _parse_issue_number(task)
    fixes_line = f"Fixes #{issue_number}" if issue_number is not None else "N/A"
    pr_body = textwrap.dedent(f"""
        ## Summary
        {task}

        ## Changes
        {clean_impl[:1000]}

        ## Testing
        {review_summary}

        _Opened by multi-agent orchestrator._

        ## Fixes
    """).strip()
    # Append at column 0 so GitHub's closing-reference scanner finds it.
    pr_body += f"\n{fixes_line}"

    from orchestrator.tools import create_pr
    import time
    branch = "ai/" + re.sub(r"[^a-z0-9]+", "-", task[:50].lower()).strip("-") + f"-{int(time.time())}"
    result = create_pr(task[:72], pr_body, branch, cwd=repo_path)
    log(result)
    return "github.com" in result or "pull" in result.lower()


def orchestrate_groom(
    task: str,
    repo_path: str,
    max_debate_rounds: int = 3,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    verbose: bool = True,
    api_key: str | None = None,
) -> bool:
    """Run backlog grooming via the GROOMER + GROOM_REVIEWERS debate loop.

    Unlike ``orchestrate``, this mode does not create a PR.  Instead the
    GROOMER applies issue management actions (label, comment, close) directly
    via the ``gh`` CLI tools, and the GROOM_REVIEWERS challenge those
    decisions.  If consensus is reached the function returns ``True``; if the
    debate loop exhausts ``max_debate_rounds`` without consensus it returns
    ``False``.

    Args:
        task:               Natural-language description of the grooming session.
        repo_path:          Absolute path to the target git repository.
        max_debate_rounds:  Maximum groomer <-> reviewer debate iterations.
        max_tool_rounds:    Hard cap on tool-call iterations per agent call.
                            Defaults to DEFAULT_MAX_TOOL_ROUNDS (100).
        verbose:            Stream progress to stdout.
        api_key:            Optional LiteLLM API key.
    """

    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    # -- Phase 1: Grooming ----------------------------------------------------
    log("\n=== GROOMER ===")
    groomer_task = task
    try:
        groom_text, groom_history = A.run(
            GROOMER,
            [{"role": "user", "content": groomer_task}],
            cwd=repo_path,
            max_tool_rounds=max_tool_rounds,
            verbose=verbose,
            api_key=api_key,
        )
    except MaxToolRoundsError as exc:
        log(f"\n=== GROOMER ABORTED: {exc} ===")
        log("Aborting run — groomer hit tool-round cap; grooming is incomplete.")
        return False
    log(f"[groomer] {groom_text[:300]}...")

    # Shared context for reviewers (without the groomer system prompt)
    shared_history = groom_history[1:]

    # -- Phase 2 + 3: Review / debate loop ------------------------------------
    verdicts: dict[str, tuple[bool, str]] = {}

    for debate_round in range(max_debate_rounds + 1):
        if debate_round == 0:
            log("\n=== INITIAL GROOM REVIEW ===")
        else:
            log(f"\n=== GROOM DEBATE ROUND {debate_round} ===")

        verdicts = {}
        reviewer_histories: dict[str, list[dict]] = {}

        for reviewer in GROOM_REVIEWERS:
            log(f"\n-- {reviewer['name']} --")
            prompt = (
                "Please review the groomer's proposed actions and give your verdict."
                if debate_round == 0
                else "The groomer has revised their actions. Please re-evaluate your previous objections."
            )
            review_text, rev_history = A.run(
                reviewer,
                shared_history + [{"role": "user", "content": prompt}],
                cwd=repo_path,
                max_tool_rounds=max_tool_rounds,
                verbose=verbose,
                api_key=api_key,
            )
            log(f"[{reviewer['name']}] {review_text[:300]}")
            approved, objection = _extract_verdict(review_text)
            verdicts[reviewer["name"]] = (approved, objection)
            reviewer_histories[reviewer["name"]] = rev_history

        objectors = {name: obj for name, (ok, obj) in verdicts.items() if not ok}

        if not objectors:
            log("\n=== GROOM CONSENSUS REACHED ===")
            break

        if debate_round == max_debate_rounds:
            log("\n=== GROOM FAILED TO REACH CONSENSUS ===")
            for name, obj in objectors.items():
                log(f"  {name}: {obj}")
            return False

        # Groomer responds to reviewer objections
        log(f"\n-- groomer rebuttal (round {debate_round + 1}) --")
        objection_summary = "\n".join(
            f"- {name}: {obj}" for name, obj in objectors.items()
        )
        rebuttal_prompt = textwrap.dedent(f"""
            The following reviewers have objections to your grooming decisions:
            {objection_summary}

            Address each concern by revising your labels/comments/closures or
            explaining why the original decision is correct.
            When done, end with: GROOMING_COMPLETE
        """).strip()

        try:
            groom_text, groom_history = A.run(
                GROOMER,
                shared_history + [{"role": "user", "content": rebuttal_prompt}],
                cwd=repo_path,
                max_tool_rounds=max_tool_rounds,
                verbose=verbose,
                api_key=api_key,
            )
        except MaxToolRoundsError as exc:
            log(f"\n=== GROOMER ABORTED (rebuttal round {debate_round + 1}): {exc} ===")
            log("Aborting run — groomer hit tool-round cap during rebuttal; grooming is incomplete.")
            return False
        log(f"[groomer] {groom_text[:300]}...")
        shared_history = groom_history[1:]  # updated shared context

    # -- Phase 4: Report grooming summary -------------------------------------
    log("\n=== GROOMING COMPLETE ===")
    review_summary = "\n".join(
        f"  {name}: {'approved' if ok else obj}"
        for name, (ok, obj) in verdicts.items()
    )
    log(f"Review summary:\n{review_summary}")
    log(f"\nGroomer final summary:\n{groom_text[:1000]}")
    return True
