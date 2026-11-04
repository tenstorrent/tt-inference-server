# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
from __future__ import annotations

import logging
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple

from workflows.bootstrap_uv import UV_EXEC
from workflows.utils import (
    get_repo_root_path,
    map_configs_by_attr,
    run_command,
)
from workflows.workflow_types import ModelType, WorkflowVenvType

if TYPE_CHECKING:
    from workflows.model_spec import ModelSpec

logger = logging.getLogger("run_log")


# Parent directory for every workflow venv; `uv venv` creates it on first use.
default_venv_path = get_repo_root_path() / ".workflow_venvs"

# Per-venv pip lists live under <repo_root>/requirements/, sharing constraints.txt.
REQUIREMENTS_DIR = get_repo_root_path() / "requirements"

# Tenstorrent fork of Harbor carrying the provider-neutral `kubernetes`
# environment (generic RKE2/EKS/... support abstracted out of what gke.py did),
# used to schedule agentic-eval trial pods on our clusters. Temporary: revert to
# harbor-framework/harbor at a release tag once the environment lands upstream.
# Track the runtime branch during development. It will be protected and require
# review before merging changes. Switch to a specific commit once development
# is stable.
HARBOR_REPO = "https://github.com/tenstorrent/harbor.git"
HARBOR_REF = "tt-inference-server"

EVALS_COMMON_LM_EVAL_COMMIT = "af401ef3dd3d2d90e55680990ed1f0c9f7892448"

# MiniMax-Provider-Verifier, run by the MiniMaxProviderVerifierTest spec test:
# its verify.py and m3_format_check pytest suites run unchanged from these
# files, and test_module/llm_tests/minimax_verifier_grading.py grades their
# output. Pinned to an upstream commit: the `verify` gate is blocking, so a
# moving branch (or a fork) must not change what is graded. This commit is
# MiniMax-AI/MiniMax-Provider-Verifier main as of #67, which carries the
# `reasoning_content` fix vLLM servers need. The `verify` cases set no
# `targets`: they are graded against the README defaults in VERIFY_METRICS
# (minimax_verifier_grading.py). The GPU vLLM measurements in MiniMax-M3's
# `_comment_threshold` (test_module/test_suites/llm.json) are reference only.
# Moving the pin can change scores: re-measure them there when it does, but
# don't add them as `targets` unless the gates are deliberately changed.
MINIMAX_VERIFIER_REPO = "https://github.com/MiniMax-AI/MiniMax-Provider-Verifier.git"
MINIMAX_VERIFIER_COMMIT = "4e778f7c3da5f0feca02741f5f736c8568760a6f"
MINIMAX_VERIFIER_DIR_NAME = "MiniMax-Provider-Verifier"
# The full tree is ~870 MB, mostly official-deployment baselines for every
# model x 10 loops. Only these paths are extracted (see export_repo_commit):
# verify.py and its sample, validator/ (imported by verify.py), the
# m3_format_check pytest suites, and the first loop of each enrolled model,
# the ToolCalls-Trigger-Similarity baseline (`verify_baseline` in
# test_module/test_suites/llm.json).
MINIMAX_VERIFIER_PATHS = (
    "verify.py",
    "sample.jsonl",
    "validator",
    "m3_format_check",
    "output-dir/MiniMax-M3/loop_01",
    "output-dir/MiniMax-M2.7/loop_01",
)
TARBALL_TIMEOUT_S = 300
# Written next to the exported files: the commit they came from.
SOURCE_COMMIT_FILE = ".source_commit"


def checkout_pinned_repo(dest: Path, repo: str, ref: str) -> bool:
    """Materialize *repo* at exactly *ref* in *dest*. Returns success.

    Idempotent, and converging rather than incremental: the directory may be
    absent, already at *ref*, left at a different revision by an earlier job,
    or not a git repository at all. Self-hosted runners keep the venv tree
    between jobs, so a "clone only when missing" shortcut would keep
    installing whatever the previous pin was.
    """
    if not (dest / ".git").is_dir():
        if dest.exists():
            logger.info("Discarding non-git directory at %s", dest)
            shutil.rmtree(dest)
        if (
            run_command(
                f"git clone --filter=blob:none --no-checkout {repo} {dest}",
                logger=logger,
            )
            != 0
        ):
            return False

    # Fetch the selected branch or commit rather than all remote refs.
    # --depth 1 keeps repeated updates from accumulating history.
    steps = (
        f"git -C {dest} remote set-url origin {repo}",
        f"git -C {dest} fetch --depth 1 origin {ref}",
        f"git -C {dest} checkout --detach --force FETCH_HEAD",
    )
    for step in steps:
        if run_command(step, logger=logger) != 0:
            logger.error("Failed to pin %s to %s (%s)", dest, ref, step)
            return False
    return True


def read_source_commit(dest: Path) -> Optional[str]:
    """The commit :func:`export_repo_commit` exported into *dest*, if any."""
    try:
        return (dest / SOURCE_COMMIT_FILE).read_text().strip() or None
    except OSError:
        return None


def tarball_url(repo: str, commit: str) -> str:
    """codeload URL of *repo*'s (a github.com URL) tarball at *commit*."""
    slug = repo.removeprefix("https://github.com/").removesuffix(".git")
    return f"https://codeload.github.com/{slug}/tar.gz/{commit}"


def _wanted_path(
    member: tarfile.TarInfo, paths: Sequence[str]
) -> Optional[PurePosixPath]:
    """*member*'s path without the tarball's top-level dir, when it is a file
    under (or equal to) one of *paths*."""
    rel = PurePosixPath(*PurePosixPath(member.name).parts[1:])
    if not member.isfile() or ".." in rel.parts:
        return None
    if any(rel == PurePosixPath(p) or PurePosixPath(p) in rel.parents for p in paths):
        return rel
    return None


def export_repo_commit(
    dest: Path, repo: str, commit: str, paths: Sequence[str]
) -> bool:
    """Materialize *paths* of *repo* at *commit* in *dest*, as plain files.
    Returns success.

    The export is re-done only when the commit recorded in ``SOURCE_COMMIT_FILE``
    is not *commit*, so a matching export needs no network at all. When the
    download fails, whatever export exists is left untouched but the call
    fails: an export of a different commit must not stand in for the pin.

    The export streams the GitHub tarball at that exact commit and extracts
    only *paths* (files, or directories taken whole) into a sibling staging
    directory, swapped into place once every path was found: no clone, so no
    ``.git`` and no dependence on the local git's sparse-checkout behaviour.
    """
    if dest.is_dir() and read_source_commit(dest) == commit:
        logger.info("%s is at %s %s", dest, repo, commit)
        return True

    url = tarball_url(repo, commit)
    staging = dest.with_name(dest.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    logger.info("Downloading %s (%s) ...", url, commit)
    try:
        with urllib.request.urlopen(
            url, timeout=TARBALL_TIMEOUT_S
        ) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
            for member in tar:
                rel = _wanted_path(member, paths)
                if rel is None:
                    continue
                out = staging / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    except (urllib.error.URLError, tarfile.TarError, OSError) as e:
        logger.error("Failed to export %s %s from %s: %s", repo, commit, url, e)
        shutil.rmtree(staging, ignore_errors=True)
        return False
    missing = [p for p in paths if not (staging / p).exists()]
    if missing:
        logger.error("%s at %s has no %s", repo, commit, ", ".join(missing))
        shutil.rmtree(staging, ignore_errors=True)
        return False
    (staging / SOURCE_COMMIT_FILE).write_text(commit + "\n")
    shutil.rmtree(dest, ignore_errors=True)
    staging.rename(dest)
    logger.info("Exported %s %s to %s", repo, commit, dest)
    return True


def install_requirements(
    venv_config: "VenvConfig",  # noqa: F821
    requirements_file: str,
    overrides_file: Optional[str] = None,
) -> bool:
    """Install pip deps from requirements/<requirements_file> into the venv.

    Always passes ``--index-strategy unsafe-best-match`` so per-file
    ``--extra-index-url`` directives resolve against all configured indexes
    (e.g. PyPI + the PyTorch CPU index).

    If ``overrides_file`` is given, it is passed to uv as ``--override`` so we
    can force a dependency version that conflicts with a package's declared
    pin (e.g. bumping transformers past vllm's ``transformers<5`` cap so the
    Gemma-4 tokenizer loads). See https://docs.astral.sh/uv/pip/compile/#overrides.
    """
    requirements_path = REQUIREMENTS_DIR / requirements_file
    if not requirements_path.is_file():
        raise FileNotFoundError(
            f"Requirements file not found: {requirements_path}. "
            f"Expected one of the per-venv files under {REQUIREMENTS_DIR}."
        )
    override_arg = ""
    if overrides_file is not None:
        overrides_path = REQUIREMENTS_DIR / overrides_file
        if not overrides_path.is_file():
            raise FileNotFoundError(
                f"Overrides file not found: {overrides_path}. "
                f"Expected one of the per-venv files under {REQUIREMENTS_DIR}."
            )
        override_arg = f"--override {overrides_path} "
    return_code = run_command(
        f"{UV_EXEC} pip install --managed-python "
        f"--python {venv_config.venv_python} "
        f"--index-strategy unsafe-best-match "
        f"{override_arg}"
        f"-r {requirements_path}",
        logger=logger,
    )
    return return_code == 0


@dataclass(frozen=True)
class VenvConfig:
    """Declarative description of a workflow virtual environment.

    ``setup()`` runs in fixed order: ``uv venv`` → mkdir ``extra_dirs`` →
    ``install_requirements(requirements_file)`` → ``setup_function`` hook.
    """

    venv_type: WorkflowVenvType
    requirements_file: Optional[str] = None
    overrides_file: Optional[str] = None
    extra_dirs: Tuple[str, ...] = field(default_factory=tuple)
    setup_function: Optional[Callable[["VenvConfig", "ModelSpec"], bool]] = None
    name: Optional[str] = None
    python_version: Optional[str] = "3.10"
    venv_path: Optional[Path] = None
    venv_python: Optional[Path] = None
    venv_pip: Optional[Path] = None

    def __post_init__(self):
        self.validate_data()
        self._infer_data()

    def validate_data(self):
        pass

    def _infer_data(self):
        if self.name is None:
            object.__setattr__(self, "name", self.venv_type.name.lower())

        if self.venv_path is None:
            object.__setattr__(
                self, "venv_path", default_venv_path / f".venv_{self.name}"
            )

        if self.venv_python is None:
            object.__setattr__(self, "venv_python", self.venv_path / "bin" / "python")

        if self.venv_pip is None:
            object.__setattr__(self, "venv_pip", self.venv_path / "bin" / "pip")

    def setup(self, model_spec: "ModelSpec") -> bool:
        """Create the venv (if missing) and install/configure it.

        Raises ``RuntimeError`` if any step fails.
        """
        if not self.venv_path.exists():
            # https://docs.astral.sh/uv/reference/cli/#uv-venv
            run_command(
                f"{str(UV_EXEC)} venv --managed-python --python={self.python_version} {self.venv_path} --allow-existing",
                logger=logger,
                check=True,
            )

        for sub_dir in self.extra_dirs:
            target = self.venv_path / sub_dir
            if target.exists():
                logger.info(f"sub-dir already exists for {self.name}: {target}")
            else:
                logger.info(f"creating sub-dir for {self.name}: {target}")
                target.mkdir(parents=True, exist_ok=True)

        if self.requirements_file is not None:
            if not install_requirements(
                self, self.requirements_file, self.overrides_file
            ):
                raise RuntimeError(
                    f"Failed to install requirements for venv {self.venv_type.name} "
                    f"from {self.requirements_file}"
                )

        if self.setup_function is not None:
            if not self.setup_function(self, model_spec=model_spec):
                raise RuntimeError(f"Failed to setup venv: {self.venv_type.name}")

        return True


def setup_evals_agentic(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    """Hook for EVALS_AGENTIC: clone + editable-install Harbor.

    Harbor is the only agentic harness on the host: it acquires the tasks,
    sandboxes them, runs the agent, and scores. The agents themselves (e.g.
    mini-swe-agent) are installed by Harbor *inside* the task container, so
    they are deliberately absent from requirements/evals-agentic.txt.

    Harbor is cloned and installed editable so its top-level ``adapters/`` directory
    is available on disk. The adapters are not part of the Harbor wheel and live
    outside ``src/``, so a ``.pth`` file exposes the repo root to Python imports.
    """
    harbor_dir = venv_config.venv_path / "harbor"
    if not checkout_pinned_repo(harbor_dir, HARBOR_REPO, HARBOR_REF):
        return False

    # Install with the `kubernetes` extra so the Python k8s client comes in: the
    # kubernetes environment needs it, and it is declared as an optional extra,
    # so a plain editable install omits it. Quoted so the shell does not
    # glob-expand `[kubernetes]`.
    return_code = run_command(
        f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} "
        f"-e '{harbor_dir}[kubernetes]'",
        logger=logger,
    )
    if return_code != 0:
        return False

    return _write_harbor_adapters_pth(venv_config, harbor_dir)


def _write_harbor_adapters_pth(
    venv_config: VenvConfig,
    harbor_dir: Path,
) -> bool:
    site_packages = next(
        (venv_config.venv_path / "lib").glob("python*/site-packages"), None
    )
    if site_packages is None:
        logger.error(
            "Could not locate site-packages under %s to write harbor-adapters.pth",
            venv_config.venv_path,
        )
        return False
    pth_file = site_packages / "harbor-adapters.pth"
    pth_file.write_text(f"{harbor_dir}\n")
    logger.info("Wrote %s pointing to %s", pth_file, harbor_dir)
    return True


def setup_minimax_verifier(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    """Hook for MINIMAX_VERIFIER: export MiniMax-Provider-Verifier at its pinned commit.

    Nothing is installed from the checkout: the spec test runs its
    ``verify.py`` and pytest suites with this venv's python, and the
    dependencies come from requirements/minimax-verifier.txt.
    """
    return export_repo_commit(
        venv_config.venv_path / MINIMAX_VERIFIER_DIR_NAME,
        MINIMAX_VERIFIER_REPO,
        MINIMAX_VERIFIER_COMMIT,
        MINIMAX_VERIFIER_PATHS,
    )


def check_docker_available(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",
) -> bool:
    """Hook for BENCHMARKS_GENAI_PERF: assert ``docker --version`` succeeds."""
    run_command("docker --version", logger=logger, check=True)
    return True


# Checkouts left by earlier setups that installed the agentic-traces client
# editable from a clone: InferenceX (which vendored it as utils/aiperf), then
# agentx-harness itself.
_LEGACY_AGENTX_CHECKOUTS = ("InferenceX", "agentx-harness")
_LEGACY_AGENTX_STAMPS = (".inferencex_ref", ".agentx_harness_ref")


def remove_legacy_agentx_checkout(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",  # noqa: F821
) -> bool:
    """Hook for AGENTIC_TRACES: drop a clone-based client install, once.

    agentx-harness now comes from PyPI via requirements/agentic-traces.txt. A
    venv set up before that still has the editable ``aiperf`` from a clone,
    which ships the same package and would shadow the release. Uninstall both
    and reinstall the requirements, since the editable dist and the release
    share files such as bin/aiperf.
    """
    legacy_dirs = [
        venv_config.venv_path / name
        for name in _LEGACY_AGENTX_CHECKOUTS
        if (venv_config.venv_path / name).is_dir()
    ]
    if not legacy_dirs:
        return True
    logger.info(
        "Replacing the clone-based agentx-harness install in %s", venv_config.venv_path
    )
    uninstall_cmd = (
        f"{UV_EXEC} pip uninstall --python {venv_config.venv_python} "
        "aiperf agentx-harness"
    )
    if run_command(uninstall_cmd, logger=logger) != 0:
        return False
    for legacy_dir in legacy_dirs:
        shutil.rmtree(legacy_dir)
    for name in _LEGACY_AGENTX_STAMPS:
        (venv_config.venv_path / name).unlink(missing_ok=True)
    return install_requirements(venv_config, venv_config.requirements_file)


def setup_evals_meta(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",
) -> bool:
    """Hook for EVALS_META: clone llama-cookbook (LLM only) and prep datasets.

    Non-LLM model types reuse this venv only for ``work_dir`` placement.
    """
    if (
        model_spec.model_type == ModelType.AUDIO
        or model_spec.model_type == ModelType.CNN
        or model_spec.model_type == ModelType.IMAGE
        or model_spec.model_type == ModelType.EMBEDDING
        or model_spec.model_type == ModelType.TEXT_TO_SPEECH
    ):
        return True

    # Cookbook's editable install and IFEval need pkg_resources. Repair cached
    # environments too, before any cookbook import or editable install.
    if (
        run_command(
            [
                str(UV_EXEC),
                "pip",
                "install",
                "--managed-python",
                "--python",
                str(venv_config.venv_python),
                "setuptools>=77,<81",
            ],
            logger=logger,
        )
        != 0
    ):
        return False
    setup_succeeded = True
    cookbook_dir = venv_config.venv_path / "llama-cookbook"
    original_dir = os.getcwd()
    if cookbook_dir.is_dir():
        logger.info(f"The directory {cookbook_dir} exists.")
    else:
        logger.info(f"The directory {cookbook_dir} does not exist. Setting up ...")
        clone_cmd = (
            f"git clone https://github.com/meta-llama/llama-cookbook.git {cookbook_dir}"
        )
        setup_succeeded = run_command(clone_cmd, logger=logger) == 0 and setup_succeeded
        setup_succeeded = (
            run_command(
                f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} -U pip",
                logger=logger,
            )
            == 0
            and setup_succeeded
        )
        # editable install is cwd-dependent, so it can't live in a requirements file
        os.chdir(cookbook_dir)
        setup_succeeded = (
            run_command(
                f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} -e .",
                logger=logger,
            )
            == 0
            and setup_succeeded
        )
        logger.warning("this might take 5 to 15+ minutes to install on first run ...")
        setup_succeeded = (
            install_requirements(venv_config, "evals-meta.txt") and setup_succeeded
        )
    # Fail before sending evaluation requests if sentence scoring cannot run.
    # Use the venv's standard NLTK data path; do not relax NLTK proxy checks.
    if (
        not setup_succeeded
        or run_command(
            [
                str(venv_config.venv_python),
                str(get_repo_root_path() / "scripts" / "setup_nltk_data.py"),
            ],
            logger=logger,
        )
        != 0
    ):
        os.chdir(original_dir)
        return False
    meta_eval_dir = (
        cookbook_dir
        / "end-to-end-use-cases"
        / "benchmarks"
        / "llm_eval_harness"
        / "meta_eval"
    )
    meta_eval_data_dir = meta_eval_dir / f"work_dir_{model_spec.model_name}"
    if not meta_eval_data_dir.exists():
        # PyYAML is only needed by this meta-eval setup hook, not by every
        # caller of ``workflow_venvs``.
        import yaml

        logger.info(f"preparing meta eval datasets for: {meta_eval_data_dir}")
        os.chdir(meta_eval_dir)
        yaml_path = meta_eval_dir / "eval_config.yaml"
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        # 3.3 reuses 3.1 evals; vision variants fall back to 3.2-3B
        _model_name = model_spec.hf_model_repo
        if _model_name == "meta-llama/Llama-3.2-11B-Vision-Instruct":
            _model_name = _model_name.replace("-3.2-11B-Vision-", "-3.2-3B-")
        elif _model_name == "meta-llama/Llama-3.2-90B-Vision-Instruct":
            _model_name = _model_name.replace("-3.2-90B-Vision-", "-3.2-3B-")
        _model_name = _model_name.replace("-3.3-", "-3.1-")
        logger.info(f"model_name: {_model_name}")

        config["work_dir"] = str(meta_eval_data_dir)
        config["model_name"] = _model_name
        config["evals_dataset"] = f"{_model_name}-evals"

        with open(yaml_path, "w") as f:
            yaml.safe_dump(config, f)

        # requires HF AUTH
        return_code = run_command(
            f"{venv_config.venv_python} prepare_meta_eval.py --config_path ./eval_config.yaml",
            logger=logger,
        )
        if return_code != 0:
            logger.warning(
                f"Failed to prepare meta eval datasets for: {meta_eval_data_dir}, continuing..."
            )
    # The model-specific data lives at meta_eval_data_dir (work_dir_<model_name>/).
    # IFEval (and likely others) hard-code ./work_dir relative to lm-eval's cwd,
    # so run_evals.py creates a per-PID staging dir with a 'work_dir' symlink
    # pointing here at command-build time. We do NOT write to a shared
    # .venv_evals_meta/work_dir/ here — that previously raced across parallel
    # model invocations and produced spurious FileNotFoundError for tasks (e.g.
    # meta_ifeval) when a sibling model's data overwrote the shared dir.
    os.chdir(original_dir)
    return setup_succeeded


def verify_evals_common_lm_eval(
    venv_config: VenvConfig,
    model_spec: ModelSpec,
) -> bool:
    """Fail closed unless EVALS_COMMON installed the reviewed harness commit.

    PEP 610 requires direct-URL installs to retain their resolved VCS commit in
    ``direct_url.json``. Printing that record puts the immutable dependency in
    every setup log; comparing it here prevents a stale persistent venv or an
    installer regression from silently running a different harness.
    """
    del model_spec
    probe = f"""
import importlib.metadata
import json

expected = {EVALS_COMMON_LM_EVAL_COMMIT!r}
dist = importlib.metadata.distribution("lm-eval")
record = json.loads(dist.read_text("direct_url.json"))
actual = record.get("vcs_info", {{}}).get("commit_id")
print(json.dumps({{"distribution": dist.metadata["Name"], "commit_id": actual}}, sort_keys=True))
if actual != expected:
    raise SystemExit(f"lm-eval commit mismatch: expected {{expected}}, installed {{actual}}")
"""
    return (
        run_command(
            [str(venv_config.venv_python), "-c", probe],
            logger=logger,
        )
        == 0
    )


def setup_evals_common(
    venv_config: VenvConfig,
    model_spec: ModelSpec,
) -> bool:
    """Verify the pinned harness, then stage the NLTK Punkt data IFEval scores with.

    ifeval / leaderboard_ifeval run in this venv, and NLTK's own downloader is
    refused by the runner proxy ("Security Violation" loading punkt_tab), so
    scoring failed after every response had been generated. Same staging step
    setup_evals_meta runs (ported from #5234).
    """
    if not verify_evals_common_lm_eval(venv_config, model_spec):
        return False
    return (
        run_command(
            [
                str(venv_config.venv_python),
                str(get_repo_root_path() / "scripts" / "setup_nltk_data.py"),
            ],
            logger=logger,
        )
        == 0
    )


# Pinned vLLM tags for the benchmark client venvs. Each must match the vllm==
# pin in its requirements file (structured-output scripts are fetched from
# vllm-project/vllm@v<pin>/benchmarks at setup time):
#   VLLM_PIN_VERSION       <-> requirements/benchmarks-vllm.txt
#   FORGE_VLLM_PIN_VERSION <-> requirements/benchmarks-vllm-forge.txt
VLLM_PIN_VERSION = "0.13.0"
FORGE_VLLM_PIN_VERSION = "0.19.1"


def _vllm_benchmarks_raw_base(pin_version: str) -> str:
    return (
        f"https://raw.githubusercontent.com/vllm-project/vllm/v{pin_version}/benchmarks"
    )


# (relative_path_in_vllm_repo, relative_path_in_work_dir)
STRUCTURED_OUTPUT_FETCH_FILES = (
    (
        "benchmark_serving_structured_output.py",
        "benchmark_serving_structured_output.py",
    ),
    ("backend_request_func.py", "backend_request_func.py"),
    (
        "structured_schemas/structured_schema_1.json",
        "structured_schemas/structured_schema_1.json",
    ),
)

# Filename of the structured-output benchmark script downloaded into the
# BENCHMARKS_VLLM venv work_dir by fetch_structured_output_scripts().
# Used to locate the structured-output benchmark script at run time.
STRUCTURED_OUTPUT_SCRIPT_NAME = "benchmark_serving_structured_output.py"


def _force_identity_encoding(client_path: Path) -> None:
    """Add Accept-Encoding: identity to the vendored benchmark client.

    The downloaded backend_request_func.py sends aiohttp's default headers,
    which advertise gzip. Gateways (e.g. console.tenstorrent.com) compress SSE
    for gzip-accepting clients and buffer each response until generation
    completes, so every chunk arrives at once and TTFT/TPOT/ITL are garbage.
    The script has no --header passthrough, so patch the headers dict instead.
    """
    text = client_path.read_text()
    if '"Accept-Encoding"' in text:
        return
    anchor = '"Content-Type": "application/json",'
    patched = text.replace(
        anchor, anchor + '\n            "Accept-Encoding": "identity",'
    )
    if patched == text:
        logger.warning(
            f"could not patch Accept-Encoding into {client_path}; "
            "streaming metrics may be invalid behind compressing gateways"
        )
        return
    client_path.write_text(patched)


def _fetch_structured_output_scripts(
    venv_config: "VenvConfig",
    pin_version: str,
) -> bool:
    """Fetch the structured-output benchmark driver scripts for ``pin_version``.

    They aren't published on PyPI, so they're pulled from the matching vLLM
    source tag at venv setup time rather than vendored into this repo.
    """
    work_dir = venv_config.venv_path / "work_dir"
    raw_base = _vllm_benchmarks_raw_base(pin_version)
    for src_rel, dst_rel in STRUCTURED_OUTPUT_FETCH_FILES:
        dst = work_dir / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{raw_base}/{src_rel}"
        return_code = run_command(
            f"curl -fSL --retry 3 --retry-delay 5 --retry-connrefused {url} -o {dst}",
            logger=logger,
        )
        if return_code != 0:
            return False
        if dst_rel == "backend_request_func.py":
            _force_identity_encoding(dst)
    return True


def fetch_structured_output_scripts(
    venv_config: "VenvConfig",
    model_spec: "ModelSpec",
) -> bool:
    """Hook for BENCHMARKS_VLLM: fetch scripts pinned to VLLM_PIN_VERSION."""
    logger.info("running fetch_structured_output_scripts() ...")
    return _fetch_structured_output_scripts(venv_config, VLLM_PIN_VERSION)


def fetch_structured_output_scripts_forge(
    venv_config: "VenvConfig",
    model_spec: "ModelSpec",
) -> bool:
    """Hook for BENCHMARKS_VLLM_FORGE: fetch scripts pinned to FORGE_VLLM_PIN_VERSION."""
    logger.info("running fetch_structured_output_scripts_forge() ...")
    return _fetch_structured_output_scripts(venv_config, FORGE_VLLM_PIN_VERSION)


_venv_config_list = [
    # Pure pip install
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_RUN_SCRIPT,
        requirements_file="evals-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.STRESS_TESTS_RUN_SCRIPT,
        requirements_file="stress-tests-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.STRESS_TESTS,
        requirements_file="stress-tests-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_RUN_SCRIPT,
        requirements_file="benchmarks-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.TESTS_RUN_SCRIPT,
        requirements_file="tests-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_COMMON,
        requirements_file="evals-common.txt",
        setup_function=setup_evals_common,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_VISION,
        requirements_file="evals-vision.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_AUDIO,
        requirements_file="evals-audio.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_EMBEDDING,
        requirements_file="evals-embedding.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.REPORTS_RUN_SCRIPT,
        requirements_file="reports-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_AGENTIC,
        requirements_file="evals-agentic.txt",
        python_version="3.12",
        setup_function=setup_evals_agentic,
    ),
    # 3.12: the verifier's pyproject requires >=3.12.
    VenvConfig(
        venv_type=WorkflowVenvType.MINIMAX_VERIFIER,
        requirements_file="minimax-verifier.txt",
        python_version="3.12",
        setup_function=setup_minimax_verifier,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.WORKFLOW_RUN_SCRIPT,
        requirements_file="workflow-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.PREFIX_CACHE,
        requirements_file="prefix-cache.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    # 3.12: the agentx-harness AIPerf fork requires >=3.11,<3.14 and its agentic
    # scenario is only exercised on 3.12 upstream.
    VenvConfig(
        venv_type=WorkflowVenvType.AGENTIC_TRACES,
        requirements_file="agentic-traces.txt",
        extra_dirs=("artifacts",),
        python_version="3.12",
        setup_function=remove_legacy_agentx_checkout,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.LLM_VLLM,
        requirements_file="llm-vllm.txt",
        # Force transformers 5.x past vllm==0.13.0's `transformers<5` cap so the
        # gemma-4 tokenizer loads; keeps vllm (and the bench-serve client) at
        # 0.13.0 for every other model. See llm-vllm-overrides.txt.
        overrides_file="llm-vllm-overrides.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.LLM_VLLM_TOKEN_TIMING,
        requirements_file="llm-vllm-token-timing.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.LLM_GUIDELLM,
        requirements_file="llm-guidellm.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.LLM_AIPERF,
        requirements_file="llm-aiperf.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.SPEC_DECODE,
        requirements_file="spec-decode.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.HF_SETUP,
        requirements_file="hf-setup.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.SYSTEM_SOFTWARE_VALIDATION,
        requirements_file="system-software-validation.txt",
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.TT_SMI,
        requirements_file="tt-smi.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.TT_TOPOLOGY,
        requirements_file="tt-topology.txt",
    ),
    # Pip install + sub-directory
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_VLLM,
        requirements_file="benchmarks-vllm.txt",
        extra_dirs=("work_dir",),
        python_version="3.11",
        setup_function=fetch_structured_output_scripts,
    ),
    # Forge-only benchmark client on newer vllm/transformers so forge tokenizers
    # load. benchmark_config.py routes forge-engine models here.
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_VLLM_FORGE,
        requirements_file="benchmarks-vllm-forge.txt",
        extra_dirs=("work_dir",),
        python_version="3.11",
        setup_function=fetch_structured_output_scripts_forge,
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.BENCHMARKS_GENAI_PERF,
        extra_dirs=("artifacts",),
        setup_function=check_docker_available,
    ),
    # Custom Python work; pip handled inside the hook (model-type dependent).
    # No extra_dirs — `run_evals.py` materializes a per-invocation staging
    # dir at command-build time (see EVALS_META branch in build_eval_command).
    VenvConfig(
        venv_type=WorkflowVenvType.EVALS_META,
        setup_function=setup_evals_meta,
    ),
]

VENV_CONFIGS = map_configs_by_attr(config_list=_venv_config_list, attr="venv_type")
