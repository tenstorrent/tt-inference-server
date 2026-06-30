"""
Debate-and-consensus orchestrator.

Flow:
  1. Implementer makes changes
  2. All reviewers audit in parallel (sequentially for now, easy to thread later)
  3. If any objections -> debate round: implementer rebuts/revises, reviewers re-evaluate
  4. Repeat up to max_debate_rounds
  5. If consensus -> create PR.  If not -> dump state and exit non-zero.
"""

import textwrap
from orchestrator.personas import IMPLEMENTER, REVIEWERS
import orchestrator.agent as A

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
    verbose: bool = True,
    api_key: str | None = None,
) -> bool:
    """Returns True on success (PR opened), False on failure.

    Args:
        task:               Natural-language description of the work to do.
        repo_path:          Absolute path to the target git repository.
        max_debate_rounds:  Maximum implementer <-> reviewer debate iterations.
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
    impl_text, impl_history = A.run(
        IMPLEMENTER,
        [{"role": "user", "content": task}],
        cwd=repo_path,
        verbose=verbose,
        api_key=api_key,
    )
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

        impl_text, impl_history = A.run(
            IMPLEMENTER,
            shared_history + [{"role": "user", "content": rebuttal_prompt}],
            cwd=repo_path,
            verbose=verbose,
            api_key=api_key,
        )
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
