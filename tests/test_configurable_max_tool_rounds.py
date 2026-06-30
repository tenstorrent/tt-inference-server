"""
Tests for issue #25: max_tool_rounds should be configurable per-task.

Verifies that:
  - DEFAULT_MAX_TOOL_ROUNDS is 100 (raised from the old hardcoded 40).
  - agent.run() uses DEFAULT_MAX_TOOL_ROUNDS when called without max_tool_rounds.
  - orchestrate() and orchestrate_groom() accept and forward max_tool_rounds to
    every A.run() call (initial impl/groom, each reviewer, and each rebuttal).
  - run.py CLI exposes --max-tool-rounds and passes it through.
"""

import inspect
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

FAKE_IMPLEMENTER = {"name": "implementer", "model": "m", "system": "s"}
FAKE_REVIEWERS = [{"name": "reviewer_a", "model": "m", "system": "s"}]
FAKE_GROOMER = {"name": "groomer", "model": "m", "system": "s"}
FAKE_GROOM_REVIEWERS = [{"name": "groom_reviewer_a", "model": "m", "system": "s"}]

_SYS_MSG = [{"role": "system", "content": "s"}]


def _stub_create_pr(monkeypatch):
    """Prevent real git/gh calls by replacing create_pr with a no-op."""
    import orchestrator.orchestrator as orch
    import orchestrator.tools as tools_mod
    monkeypatch.setattr(tools_mod, "create_pr", lambda *a, **kw: "https://github.com/fake/pull/1")


# ---------------------------------------------------------------------------
# DEFAULT_MAX_TOOL_ROUNDS constant
# ---------------------------------------------------------------------------

class TestDefaultMaxToolRounds:
    def test_default_is_100(self):
        """DEFAULT_MAX_TOOL_ROUNDS must be 100 (generous safety rail)."""
        from orchestrator.agent import DEFAULT_MAX_TOOL_ROUNDS
        assert DEFAULT_MAX_TOOL_ROUNDS == 100

    def test_exported_from_package(self):
        """DEFAULT_MAX_TOOL_ROUNDS must be importable from the top-level package."""
        from orchestrator import DEFAULT_MAX_TOOL_ROUNDS
        assert DEFAULT_MAX_TOOL_ROUNDS == 100

    def test_agent_run_uses_default_when_not_specified(self):
        """agent.run() signature must default max_tool_rounds to DEFAULT_MAX_TOOL_ROUNDS."""
        from orchestrator.agent import run, DEFAULT_MAX_TOOL_ROUNDS
        sig = inspect.signature(run)
        assert sig.parameters["max_tool_rounds"].default == DEFAULT_MAX_TOOL_ROUNDS

    def test_orchestrate_uses_default_when_not_specified(self):
        """orchestrate() signature must default max_tool_rounds to DEFAULT_MAX_TOOL_ROUNDS."""
        from orchestrator.orchestrator import orchestrate
        from orchestrator.agent import DEFAULT_MAX_TOOL_ROUNDS
        sig = inspect.signature(orchestrate)
        assert sig.parameters["max_tool_rounds"].default == DEFAULT_MAX_TOOL_ROUNDS

    def test_orchestrate_groom_uses_default_when_not_specified(self):
        """orchestrate_groom() signature must default max_tool_rounds to DEFAULT_MAX_TOOL_ROUNDS."""
        from orchestrator.orchestrator import orchestrate_groom
        from orchestrator.agent import DEFAULT_MAX_TOOL_ROUNDS
        sig = inspect.signature(orchestrate_groom)
        assert sig.parameters["max_tool_rounds"].default == DEFAULT_MAX_TOOL_ROUNDS


# ---------------------------------------------------------------------------
# orchestrate() forwards max_tool_rounds to every A.run() call
# ---------------------------------------------------------------------------

class TestOrchestrateForwardsMaxToolRounds:
    def _patch_personas(self, monkeypatch):
        import orchestrator.orchestrator as orch
        monkeypatch.setattr(orch, "IMPLEMENTER", FAKE_IMPLEMENTER)
        monkeypatch.setattr(orch, "REVIEWERS", FAKE_REVIEWERS)

    def test_custom_value_forwarded_to_implementer(self, monkeypatch):
        """orchestrate() must pass the caller-supplied max_tool_rounds to the
        implementer A.run() call."""
        import orchestrator.orchestrator as orch
        self._patch_personas(monkeypatch)
        _stub_create_pr(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                captured["impl_max"] = kwargs.get("max_tool_rounds")
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        orch.orchestrate("task", "/fake/repo", max_tool_rounds=7, verbose=False)
        assert captured["impl_max"] == 7

    def test_custom_value_forwarded_to_reviewers(self, monkeypatch):
        """orchestrate() must pass the caller-supplied max_tool_rounds to every
        reviewer A.run() call."""
        import orchestrator.orchestrator as orch
        self._patch_personas(monkeypatch)
        _stub_create_pr(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "reviewer_a":
                captured["rev_max"] = kwargs.get("max_tool_rounds")
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        orch.orchestrate("task", "/fake/repo", max_tool_rounds=13, verbose=False)
        assert captured["rev_max"] == 13

    def test_custom_value_forwarded_to_rebuttal(self, monkeypatch):
        """orchestrate() must pass max_tool_rounds to the rebuttal A.run() call."""
        import orchestrator.orchestrator as orch
        self._patch_personas(monkeypatch)
        _stub_create_pr(monkeypatch)

        call_counts = {"impl": 0}
        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                call_counts["impl"] += 1
                if call_counts["impl"] == 1:
                    return "done", _SYS_MSG[:]
                # Second call is the rebuttal
                captured["rebuttal_max"] = kwargs.get("max_tool_rounds")
                return "IMPLEMENTATION_COMPLETE", _SYS_MSG[:]
            # Reviewer objects on first debate round, then approves
            if call_counts["impl"] == 1:
                return "OBJECTION: something", _SYS_MSG[:]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        orch.orchestrate("task", "/fake/repo", max_tool_rounds=42, max_debate_rounds=3, verbose=False)
        assert captured.get("rebuttal_max") == 42

    def test_all_run_calls_receive_same_max_tool_rounds(self, monkeypatch):
        """Every single A.run() call inside orchestrate() must receive the same
        max_tool_rounds value that was passed to orchestrate()."""
        import orchestrator.orchestrator as orch
        self._patch_personas(monkeypatch)
        _stub_create_pr(monkeypatch)

        seen_values = []
        call_counts = {"impl": 0}

        def fake_run(persona, messages, **kwargs):
            seen_values.append(kwargs.get("max_tool_rounds"))
            if persona["name"] == "implementer":
                call_counts["impl"] += 1
                if call_counts["impl"] == 1:
                    return "done", _SYS_MSG[:]
                return "IMPLEMENTATION_COMPLETE", _SYS_MSG[:]
            # Reviewer objects once to exercise the rebuttal path, then approves
            if call_counts["impl"] == 1:
                return "OBJECTION: nope", _SYS_MSG[:]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        orch.orchestrate("task", "/fake/repo", max_tool_rounds=55, max_debate_rounds=3, verbose=False)
        assert len(seen_values) > 0
        assert all(v == 55 for v in seen_values), (
            f"Not all A.run() calls received max_tool_rounds=55: {seen_values}"
        )


# ---------------------------------------------------------------------------
# orchestrate_groom() forwards max_tool_rounds to every A.run() call
# ---------------------------------------------------------------------------

class TestOrchestrateGroomForwardsMaxToolRounds:
    def _patch_personas(self, monkeypatch):
        import orchestrator.orchestrator as orch
        monkeypatch.setattr(orch, "GROOMER", FAKE_GROOMER)
        monkeypatch.setattr(orch, "GROOM_REVIEWERS", FAKE_GROOM_REVIEWERS)

    def _stub_list_issues(self, monkeypatch):
        import orchestrator.tools as tools_mod
        monkeypatch.setattr(tools_mod, "list_issues", lambda **kw: "[]")

    def test_custom_value_forwarded_to_groomer(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch_personas(monkeypatch)
        self._stub_list_issues(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "groomer":
                captured["groom_max"] = kwargs.get("max_tool_rounds")
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        orch.orchestrate_groom("triage", "/fake/repo", max_tool_rounds=8, verbose=False)
        assert captured["groom_max"] == 8

    def test_custom_value_forwarded_to_groom_reviewers(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch_personas(monkeypatch)
        self._stub_list_issues(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "groom_reviewer_a":
                captured["rev_max"] = kwargs.get("max_tool_rounds")
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        orch.orchestrate_groom("triage", "/fake/repo", max_tool_rounds=17, verbose=False)
        assert captured["rev_max"] == 17

    def test_all_groom_run_calls_receive_same_max_tool_rounds(self, monkeypatch):
        """Every A.run() call inside orchestrate_groom() must receive the same
        max_tool_rounds that was passed to orchestrate_groom()."""
        import orchestrator.orchestrator as orch
        self._patch_personas(monkeypatch)
        self._stub_list_issues(monkeypatch)

        seen_values = []
        call_counts = {"groom": 0}

        def fake_run(persona, messages, **kwargs):
            seen_values.append(kwargs.get("max_tool_rounds"))
            if persona["name"] == "groomer":
                call_counts["groom"] += 1
                if call_counts["groom"] == 1:
                    return "GROOMING_COMPLETE", _SYS_MSG[:]
                return "GROOMING_COMPLETE", _SYS_MSG[:]
            # Reviewer objects once to exercise the rebuttal path
            if call_counts["groom"] == 1:
                return "OBJECTION: bad labels", _SYS_MSG[:]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)

        orch.orchestrate_groom("triage", "/fake/repo", max_tool_rounds=77, max_debate_rounds=3, verbose=False)
        assert len(seen_values) > 0
        assert all(v == 77 for v in seen_values), (
            f"Not all A.run() calls received max_tool_rounds=77: {seen_values}"
        )


# ---------------------------------------------------------------------------
# run.py CLI: --max-tool-rounds argument
# ---------------------------------------------------------------------------

class TestRunPyCLI:
    def _load_run_module(self):
        """Import run.py as a module without executing main()."""
        import importlib.util, os
        run_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "run.py",
        )
        spec = importlib.util.spec_from_file_location("run_module", run_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_max_tool_rounds_arg_exists(self):
        """run.py must expose --max-tool-rounds as a CLI argument."""
        mod = self._load_run_module()
        parser = mod.build_parser()
        args = parser.parse_args(["/fake/repo", "some task"])
        assert hasattr(args, "max_tool_rounds")

    def test_default_is_100(self):
        """--max-tool-rounds must default to DEFAULT_MAX_TOOL_ROUNDS (100)."""
        from orchestrator import DEFAULT_MAX_TOOL_ROUNDS
        mod = self._load_run_module()
        parser = mod.build_parser()
        args = parser.parse_args(["/fake/repo", "some task"])
        assert args.max_tool_rounds == DEFAULT_MAX_TOOL_ROUNDS

    def test_custom_value_parsed(self):
        """--max-tool-rounds N must be parsed as integer N."""
        mod = self._load_run_module()
        parser = mod.build_parser()
        args = parser.parse_args(["--max-tool-rounds", "25", "/fake/repo", "some task"])
        assert args.max_tool_rounds == 25

    def test_passed_to_orchestrate(self, monkeypatch, tmp_path):
        """main() must pass --max-tool-rounds to orchestrate()."""
        import sys
        mod = self._load_run_module()

        captured = {}

        def fake_orchestrate(task, repo_path, **kwargs):
            captured["max_tool_rounds"] = kwargs.get("max_tool_rounds")
            return True

        def fake_orchestrate_groom(task, repo_path, **kwargs):  # pragma: no cover
            return True

        # Create a fake git repo directory
        fake_repo = tmp_path / "repo"
        fake_repo.mkdir()
        (fake_repo / ".git").mkdir()

        monkeypatch.setattr(mod, "orchestrate", fake_orchestrate)
        monkeypatch.setattr(mod, "orchestrate_groom", fake_orchestrate_groom)
        monkeypatch.setattr(
            sys, "argv",
            ["run.py", "--max-tool-rounds", "33", str(fake_repo), "do something"],
        )

        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 0
        assert captured["max_tool_rounds"] == 33

    def test_passed_to_orchestrate_groom(self, monkeypatch, tmp_path):
        """main() must pass --max-tool-rounds to orchestrate_groom() in groom mode."""
        import sys
        mod = self._load_run_module()

        captured = {}

        def fake_orchestrate(task, repo_path, **kwargs):  # pragma: no cover
            return True

        def fake_orchestrate_groom(task, repo_path, **kwargs):
            captured["max_tool_rounds"] = kwargs.get("max_tool_rounds")
            return True

        fake_repo = tmp_path / "repo"
        fake_repo.mkdir()
        (fake_repo / ".git").mkdir()

        monkeypatch.setattr(mod, "orchestrate", fake_orchestrate)
        monkeypatch.setattr(mod, "orchestrate_groom", fake_orchestrate_groom)
        monkeypatch.setattr(
            sys, "argv",
            ["run.py", "--mode", "groom", "--max-tool-rounds", "50",
             str(fake_repo), "triage issues"],
        )

        with pytest.raises(SystemExit) as exc_info:
            mod.main()
        assert exc_info.value.code == 0
        assert captured["max_tool_rounds"] == 50
