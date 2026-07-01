"""
Tests for issue #131: --implementer-* and --reviewer-* CLI flags in run.py.

Covers:
  - build_parser() exposes all five new flags
  - build_implementer_override() builds the correct dict from parsed args
  - build_reviewer_override() builds the correct dict from parsed args
  - tt-console auto-applies 32768 max_tokens when no explicit value given
  - Explicit --implementer-max-tokens overrides the tt-console default
  - Omitting all override flags produces empty dicts (no-op for defaults)
  - orchestrate() accepts implementer_override and reviewer_override params
  - orchestrate() merges overrides into the implementer persona (model + provider)
  - orchestrate() merges overrides into every reviewer persona
  - orchestrate() leaves unmentioned persona fields (system, name) intact
  - Omitting overrides keeps the original persona defaults (backward compat)
  - run.py main() passes overrides through to orchestrate()
"""

import inspect
import copy
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SYS_MSG = [{"role": "system", "content": "s"}]

FAKE_IMPLEMENTER = {
    "name": "implementer",
    "model": "default-impl-model",
    "provider": "litellm",
    "system": "impl-system-prompt",
    "max_tokens": 4096,
}
FAKE_REVIEWER = {
    "name": "correctness_reviewer",
    "model": "default-rev-model",
    "provider": "litellm",
    "system": "rev-system-prompt",
}


def _make_args(**kwargs):
    """Return a namespace simulating argparse output for the new flags."""
    import argparse
    defaults = dict(
        implementer_model=None,
        implementer_provider=None,
        implementer_max_tokens=None,
        reviewer_model=None,
        reviewer_provider=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Parser: new flags are registered
# ---------------------------------------------------------------------------

class TestParserFlags:
    def _parser(self):
        import importlib, sys
        # Re-import to get a fresh parser each time
        import run as run_mod
        return run_mod.build_parser()

    def test_implementer_model_flag_exists(self):
        p = self._parser()
        args = p.parse_args(["--implementer-model", "foo/bar", ".", "task"])
        assert args.implementer_model == "foo/bar"

    def test_implementer_provider_flag_exists(self):
        p = self._parser()
        args = p.parse_args(["--implementer-provider", "tt-console", ".", "task"])
        assert args.implementer_provider == "tt-console"

    def test_implementer_max_tokens_flag_exists(self):
        p = self._parser()
        args = p.parse_args(["--implementer-max-tokens", "8192", ".", "task"])
        assert args.implementer_max_tokens == 8192

    def test_reviewer_model_flag_exists(self):
        p = self._parser()
        args = p.parse_args(["--reviewer-model", "foo/rev", ".", "task"])
        assert args.reviewer_model == "foo/rev"

    def test_reviewer_provider_flag_exists(self):
        p = self._parser()
        args = p.parse_args(["--reviewer-provider", "litellm", ".", "task"])
        assert args.reviewer_provider == "litellm"

    def test_all_flags_default_to_none(self):
        p = self._parser()
        args = p.parse_args([".", "task"])
        assert args.implementer_model is None
        assert args.implementer_provider is None
        assert args.implementer_max_tokens is None
        assert args.reviewer_model is None
        assert args.reviewer_provider is None

    def test_implementer_max_tokens_is_int(self):
        p = self._parser()
        args = p.parse_args(["--implementer-max-tokens", "16384", ".", "task"])
        assert isinstance(args.implementer_max_tokens, int)


# ---------------------------------------------------------------------------
# build_implementer_override()
# ---------------------------------------------------------------------------

class TestBuildImplementerOverride:
    def _call(self, **kwargs):
        import run as run_mod
        return run_mod.build_implementer_override(_make_args(**kwargs))

    def test_empty_when_no_flags(self):
        assert self._call() == {}

    def test_model_only(self):
        result = self._call(implementer_model="x/y")
        assert result == {"model": "x/y"}

    def test_provider_only(self):
        result = self._call(implementer_provider="litellm")
        assert result == {"provider": "litellm"}

    def test_model_and_provider(self):
        result = self._call(implementer_model="a/b", implementer_provider="tt-console")
        assert result["model"] == "a/b"
        assert result["provider"] == "tt-console"

    def test_explicit_max_tokens(self):
        result = self._call(implementer_max_tokens=8192)
        assert result["max_tokens"] == 8192

    def test_tt_console_default_max_tokens(self):
        # When provider is tt-console and no explicit max_tokens, default to 32768
        import run as run_mod
        result = self._call(implementer_provider="tt-console")
        assert result["max_tokens"] == run_mod._TT_CONSOLE_DEFAULT_MAX_TOKENS

    def test_explicit_max_tokens_beats_tt_console_default(self):
        result = self._call(implementer_provider="tt-console", implementer_max_tokens=1234)
        assert result["max_tokens"] == 1234

    def test_litellm_provider_no_default_max_tokens(self):
        result = self._call(implementer_provider="litellm")
        assert "max_tokens" not in result


# ---------------------------------------------------------------------------
# build_reviewer_override()
# ---------------------------------------------------------------------------

class TestBuildReviewerOverride:
    def _call(self, **kwargs):
        import run as run_mod
        return run_mod.build_reviewer_override(_make_args(**kwargs))

    def test_empty_when_no_flags(self):
        assert self._call() == {}

    def test_model_only(self):
        assert self._call(reviewer_model="r/m") == {"model": "r/m"}

    def test_provider_only(self):
        assert self._call(reviewer_provider="tt-console") == {"provider": "tt-console"}

    def test_model_and_provider(self):
        result = self._call(reviewer_model="r/m", reviewer_provider="litellm")
        assert result == {"model": "r/m", "provider": "litellm"}


# ---------------------------------------------------------------------------
# orchestrate() signature: new params with None defaults
# ---------------------------------------------------------------------------

class TestOrchestrateSignature:
    def test_implementer_override_param_exists(self):
        from orchestrator.orchestrator import orchestrate
        sig = inspect.signature(orchestrate)
        assert "implementer_override" in sig.parameters

    def test_reviewer_override_param_exists(self):
        from orchestrator.orchestrator import orchestrate
        sig = inspect.signature(orchestrate)
        assert "reviewer_override" in sig.parameters

    def test_implementer_override_defaults_to_none(self):
        from orchestrator.orchestrator import orchestrate
        sig = inspect.signature(orchestrate)
        assert sig.parameters["implementer_override"].default is None

    def test_reviewer_override_defaults_to_none(self):
        from orchestrator.orchestrator import orchestrate
        sig = inspect.signature(orchestrate)
        assert sig.parameters["reviewer_override"].default is None


# ---------------------------------------------------------------------------
# orchestrate() applies implementer_override correctly
# ---------------------------------------------------------------------------

class TestOrchestrateImplementerOverride:
    def _patch(self, monkeypatch):
        import orchestrator.orchestrator as orch
        monkeypatch.setattr(orch, "IMPLEMENTER", copy.deepcopy(FAKE_IMPLEMENTER))
        monkeypatch.setattr(orch, "REVIEWERS", [copy.deepcopy(FAKE_REVIEWER)])
        monkeypatch.setattr(
            "orchestrator.tools.create_pr",
            lambda *a, **kw: "https://github.com/fake/pull/1",
        )

    def test_model_overridden_on_implementer(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                captured["model"] = persona["model"]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("task", "/fake", verbose=False,
                         implementer_override={"model": "new-model"})
        assert captured["model"] == "new-model"

    def test_provider_overridden_on_implementer(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                captured["provider"] = persona["provider"]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("task", "/fake", verbose=False,
                         implementer_override={"provider": "tt-console"})
        assert captured["provider"] == "tt-console"

    def test_system_prompt_preserved_after_override(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                captured["system"] = persona.get("system")
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("task", "/fake", verbose=False,
                         implementer_override={"model": "x"})
        assert captured["system"] == FAKE_IMPLEMENTER["system"]

    def test_original_implementer_persona_not_mutated(self, monkeypatch):
        import orchestrator.orchestrator as orch
        original = copy.deepcopy(FAKE_IMPLEMENTER)
        self._patch(monkeypatch)
        monkeypatch.setattr(orch, "IMPLEMENTER", original)

        def fake_run(persona, messages, **kwargs):
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("task", "/fake", verbose=False,
                         implementer_override={"model": "mutant-model"})
        assert original["model"] == FAKE_IMPLEMENTER["model"]

    def test_no_override_uses_default_model(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                captured["model"] = persona["model"]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("task", "/fake", verbose=False)
        assert captured["model"] == FAKE_IMPLEMENTER["model"]


# ---------------------------------------------------------------------------
# orchestrate() applies reviewer_override to ALL reviewers
# ---------------------------------------------------------------------------

class TestOrchestrateReviewerOverride:
    def _patch_two_reviewers(self, monkeypatch):
        import orchestrator.orchestrator as orch
        reviewer_b = {**FAKE_REVIEWER, "name": "security_reviewer"}
        monkeypatch.setattr(orch, "IMPLEMENTER", copy.deepcopy(FAKE_IMPLEMENTER))
        monkeypatch.setattr(orch, "REVIEWERS", [
            copy.deepcopy(FAKE_REVIEWER),
            copy.deepcopy(reviewer_b),
        ])
        monkeypatch.setattr(
            "orchestrator.tools.create_pr",
            lambda *a, **kw: "https://github.com/fake/pull/1",
        )

    def test_model_overridden_on_all_reviewers(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch_two_reviewers(monkeypatch)

        captured = []

        def fake_run(persona, messages, **kwargs):
            if persona["name"] != "implementer":
                captured.append(persona["model"])
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("task", "/fake", verbose=False,
                         reviewer_override={"model": "rev-override"})
        assert len(captured) == 2
        assert all(m == "rev-override" for m in captured)

    def test_system_prompt_preserved_on_reviewers(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch_two_reviewers(monkeypatch)

        captured = []

        def fake_run(persona, messages, **kwargs):
            if persona["name"] != "implementer":
                captured.append(persona.get("system"))
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("task", "/fake", verbose=False,
                         reviewer_override={"model": "x"})
        assert all(s == FAKE_REVIEWER["system"] for s in captured)

    def test_reviewer_override_does_not_affect_implementer(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch_two_reviewers(monkeypatch)

        captured = {}

        def fake_run(persona, messages, **kwargs):
            if persona["name"] == "implementer":
                captured["model"] = persona["model"]
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("task", "/fake", verbose=False,
                         reviewer_override={"model": "rev-model"})
        assert captured["model"] == FAKE_IMPLEMENTER["model"]

    def test_no_reviewer_override_keeps_defaults(self, monkeypatch):
        import orchestrator.orchestrator as orch
        self._patch_two_reviewers(monkeypatch)

        captured = []

        def fake_run(persona, messages, **kwargs):
            if persona["name"] != "implementer":
                captured.append(persona["model"])
            return "APPROVED", _SYS_MSG[:]

        monkeypatch.setattr(orch.A, "run", fake_run)
        orch.orchestrate("task", "/fake", verbose=False)
        assert all(m == FAKE_REVIEWER["model"] for m in captured)


# ---------------------------------------------------------------------------
# run.py main() passes overrides to orchestrate()
# ---------------------------------------------------------------------------

class TestMainPassesOverrides:
    def _run_main(self, monkeypatch, tmp_path, extra_args, captured):
        import run as run_mod
        import orchestrator.orchestrator as orch

        git_dir = tmp_path / ".git"
        git_dir.mkdir()

        monkeypatch.setattr(
            "orchestrator.config.validate_provider_keys", lambda providers: None
        )
        monkeypatch.setattr(
            "subprocess.run",
            lambda *a, **kw: MagicMock(returncode=0, stdout="", stderr=""),
        )
        monkeypatch.setattr(run_mod, "assign_issue_if_present", lambda *a: None)
        monkeypatch.setattr(run_mod, "write_status", lambda *a, **kw: None)

        def fake_orchestrate(*a, **kw):
            captured.update(kw)
            return True

        monkeypatch.setattr(run_mod, "orchestrate", fake_orchestrate)

        argv = extra_args + [str(tmp_path), "task text"]
        monkeypatch.setattr("sys.argv", ["run.py"] + argv)

        try:
            run_mod.main()
        except SystemExit:
            pass

    def test_implementer_override_forwarded(self, monkeypatch, tmp_path):
        captured = {}
        self._run_main(
            monkeypatch, tmp_path,
            ["--implementer-model", "kimi/x", "--implementer-provider", "tt-console"],
            captured,
        )
        override = captured.get("implementer_override") or {}
        assert override.get("model") == "kimi/x"
        assert override.get("provider") == "tt-console"

    def test_reviewer_override_forwarded(self, monkeypatch, tmp_path):
        captured = {}
        self._run_main(
            monkeypatch, tmp_path,
            ["--reviewer-model", "sonnet/4", "--reviewer-provider", "litellm"],
            captured,
        )
        override = captured.get("reviewer_override") or {}
        assert override.get("model") == "sonnet/4"
        assert override.get("provider") == "litellm"

    def test_no_flags_passes_none_overrides(self, monkeypatch, tmp_path):
        captured = {}
        self._run_main(monkeypatch, tmp_path, [], captured)
        # Empty dicts are normalised to None before forwarding
        impl_ov = captured.get("implementer_override")
        rev_ov = captured.get("reviewer_override")
        assert not impl_ov
        assert not rev_ov

    def test_tt_console_default_max_tokens_forwarded(self, monkeypatch, tmp_path):
        import run as run_mod
        captured = {}
        self._run_main(
            monkeypatch, tmp_path,
            ["--implementer-provider", "tt-console"],
            captured,
        )
        override = captured.get("implementer_override") or {}
        assert override.get("max_tokens") == run_mod._TT_CONSOLE_DEFAULT_MAX_TOKENS
