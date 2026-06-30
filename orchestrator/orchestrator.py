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

import textwrap
from orchestrator.personas import (
    IMPLEMENTER, REVIEWERS,
    GROOMER, GROOM_REVIEWERS,
)
import orchestrator.agent as A
from orchestrator.agent import MaxToolRoundsError, DEFAULT_MAX_TOOL_ROUNDS


def _extract_verdict(text: str) -> tuple[bool, str]:
    """Returns (approved, objection_text).

    Scans from the bottom up so that a reviewer who says
    'my previous OBJECTION is resolved - APPROVED' counts as approved.
    """
    for line in reversed(text.splitlines()):
        stripped = line.strip().upper()
        if stripped == "APPROVED" or stripped.startswith("APPROVED"):
            return True, ""
        if stripped.startswith("OBJECTION"):
            return False, line.strip()
    # No explicit verdict line found - check if APPROVED appears anywhere at end
    if "APPROVED" in text[-300:].upper():
        return True, ""
    return False, text[-500:]


def orchestrate(
    task: str,
    repo_path: str,
    max_debate_rounds: int = 3,
    max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    verbose: bool = True,
    api_key: str | None = None,
) -> bool:
    """Returns True on success (PR opened), False on failure.

    Args:
        task:               Natural-language description of the work to do.
        repo_path:          Absolute path to the target git repository.
        max_debate_rounds:  Maximum implementer <-> reviewer debate iterations.
        max_tool_rounds:    Hard cap on tool-call iterations per agent call.
                            Defaults to DEFAULT_MAX_TOOL_ROUNDS (100).  Pass a
                            lower value for simple tasks, a higher value for
                            complex ones.
        verbose:            Stream progress to stdout.
        api_key:            Optional LiteLLM API key.  Falls back to the
                            ``TT_CHAT_API_KEY`` env-var and then the key file
                            when None.
    """

    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    # -- Phase 1: Implementation ----------------------------------------------
    log("\n=== IMPLEMENTER ===")
    try:
        impl_text, impl_history = A.run(
            IMPLEMENTER,
            [{"role": "user", "content": task}],
            cwd=repo_path,
            max_tool_rounds=max_tool_rounds,
            verbose=verbose,
            api_key=api_key,
        )
    except MaxToolRoundsError as exc:
        log(f"\n=== IMPLEMENTER ABORTED: {exc} ===")
        log("Aborting run — implementer hit tool-round cap; work is incomplete.")
        return False
    log(f"[implementer] {impl_text[:300]}...")

    # Strip the system message; shared context for reviewers starts here
    shared_history = impl_history[1:]  # drop system prompt

    # -- Phase 2 + 3: Review / debate loop ------------------------------------
    for debate_round in range(max_debate_rounds + 1):
        if debate_round == 0:
            log("\n=== INITIAL REVIEW ===")
        else:
            log(f"\n=== DEBATE ROUND {debate_round} ===")

        verdicts: dict[str, tuple[bool, str]] = {}
        reviewer_histories: dict[str, list[dict]] = {}

        for reviewer in REVIEWERS:
            log(f"\n-- {reviewer['name']} --")
            prompt = (
                "Please review the implementation and give your verdict."
                if debate_round == 0
                else "The implementer has revised. Please re-evaluate your previous objections."
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

        try:
            impl_text, impl_history = A.run(
                IMPLEMENTER,
                shared_history + [{"role": "user", "content": rebuttal_prompt}],
                cwd=repo_path,
                max_tool_rounds=max_tool_rounds,
                verbose=verbose,
                api_key=api_key,
            )
        except MaxToolRoundsError as exc:
            log(f"\n=== IMPLEMENTER ABORTED (rebuttal round {debate_round + 1}): {exc} ===")
            log("Aborting run — implementer hit tool-round cap during rebuttal; work is incomplete.")
            return False
        log(f"[implementer] {impl_text[:300]}...")
        shared_history = impl_history[1:]  # updated shared context

    # -- Phase 4: Open PR -----------------------------------------------------
    log("\n=== OPENING PR ===")
    review_summary = "\n".join(
        f"- **{name}**: {'approved' if ok else obj}"
        for name, (ok, obj) in verdicts.items()
    )
    pr_body = textwrap.dedent(f"""
        ## Task
        {task}

        ## Implementation summary
        {impl_text[:1000]}

        ## Review summary
        {review_summary}

        _Opened by multi-agent orchestrator._
    """).strip()

    from orchestrator.tools import create_pr
    import re, time
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
        task:               Natural-language description of the grooming goal
                            (e.g. "triage open issues and assign priorities").
        repo_path:          Absolute path to the target git repository.  The
                            ``gh`` CLI commands are executed in this directory
                            so that they pick up the correct GitHub remote.
        max_debate_rounds:  Maximum groomer <-> reviewer debate iterations.
        max_tool_rounds:    Hard cap on tool-call iterations per agent call.
                            Defaults to DEFAULT_MAX_TOOL_ROUNDS (100).  Pass a
                            lower value for simple tasks, a higher value for
                            complex ones.
        verbose:            Stream progress to stdout.
        api_key:            Optional LiteLLM API key.  Falls back to the
                            ``TT_CHAT_API_KEY`` env-var and then the key file
                            when None.
    """

    def log(msg: str):
        if verbose:
            print(msg, flush=True)

    # -- Fetch open issues to give the groomer rich context up-front ----------
    from orchestrator.tools import list_issues as _list_issues
    log("\n=== FETCHING OPEN ISSUES ===")
    issues_json = _list_issues(state="open", limit=200, cwd=repo_path)
    log(f"[context] fetched issue list ({len(issues_json)} chars)")

    groomer_task = textwrap.dedent(f"""
        {task}

        Here is the current list of open issues in JSON format:
        {issues_json}

        Analyse each issue and apply the appropriate labels, comments, and
        closures (for confirmed duplicates or out-of-scope items).
        When you have finished processing all issues end with: GROOMING_COMPLETE
    """).strip()

    # -- Phase 1: Grooming ----------------------------------------------------
    log("\n=== GROOMER ===")
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
