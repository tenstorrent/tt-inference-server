# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional, Tuple

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
    """Hook for EVALS_AGENTIC: clone + editable-install SWE-agent and Harbor.

    Other deps (mini-swe-agent, epoch SWE-bench) are in requirements/evals-agentic.txt.

    Harbor is cloned and installed editable so its top-level ``adapters/`` directory
    is available on disk. The adapters are not part of the Harbor wheel and live
    outside ``src/``, so a ``.pth`` file exposes the repo root to Python imports.
    """
    sweagent_dir = venv_config.venv_path / "SWE-agent"
    if not sweagent_dir.exists():
        clone_return_code = run_command(
            f"git clone https://github.com/SWE-agent/SWE-agent.git {sweagent_dir}",
            logger=logger,
        )
        if clone_return_code != 0:
            return False

    return_code = run_command(
        f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} "
        f"-e {sweagent_dir}",
        logger=logger,
    )
    if return_code != 0:
        return False

    harbor_dir = venv_config.venv_path / "harbor"
    harbor_tag = "v0.6.5"
    if not harbor_dir.exists():
        clone_return_code = run_command(
            "git clone --depth 1 --branch "
            f"{harbor_tag} https://github.com/harbor-framework/harbor.git {harbor_dir}",
            logger=logger,
        )
        if clone_return_code != 0:
            return False

    return_code = run_command(
        f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} "
        f"-e {harbor_dir}",
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


def check_docker_available(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",
) -> bool:
    """Hook for BENCHMARKS_GENAI_PERF: assert ``docker --version`` succeeds."""
    run_command("docker --version", logger=logger, check=True)
    return True


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
        # cookbook editable install needs modern setuptools
        setup_succeeded = (
            run_command(
                f"{UV_EXEC} pip install --managed-python --python {venv_config.venv_python} -U pip setuptools",
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
# Imported by benchmarking/run_benchmarks.py to locate the script at run time.
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


# Sentinel markers appended to lm-eval so the patches are applied at most once
# per venv (idempotent across repeated setup runs). Version the API sentinel so
# venvs containing the original streaming-only patch receive the reasoning
# preservation update too.
_LM_EVAL_CHAT_STREAM_SENTINEL = (
    "# === TT patch: chat-completions SSE streaming + reasoning v2 ==="
)
_LM_EVAL_REASONING_LOG_SENTINEL = (
    "# === TT patch: preserve chat reasoning in sample logs ==="
)

# Monkeypatch appended to the END of lm_eval/models/api_models.py (after
# TemplateAPI is defined). Stock _consume_sse_stream only handled text-completion
# chunks (choice["text"]) and emitted {"index","text"}. Chat-completions stream
# tokens in choice["delta"]["content"] and the chat parser reads
# choice["message"]["content"], so streamed chat responses raised
# KeyError: 'message'. This makes streaming handle both shapes, teaches the
# synchronous model_call() path to consume SSE (stock sync path had none), and
# carries reasoning_content alongside the final answer without exposing it to
# task filters or metrics.
_LM_EVAL_CHAT_STREAM_PATCH = """

# === TT patch: chat-completions SSE streaming + reasoning v2 ===
# Applied post-install by workflows.workflow_venvs.patch_evals_common_chat_streaming.
class _TTChatGeneration(str):
    def __new__(cls, content: str, reasoning_content: str):
        value = super().__new__(cls, content)
        value.reasoning_content = reasoning_content
        return value


def _tt_wrap_chat_response_reasoning(response: dict) -> dict:
    for choice in response.get("choices", []):
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        if "reasoning_content" in message:
            reasoning = message.get("reasoning_content")
        elif "reasoning" in message:
            reasoning = message.get("reasoning")
        else:
            continue
        message["content"] = _TTChatGeneration(
            message.get("content") or "",
            reasoning or "",
        )
    return response


def _tt_stream_chunk_parts(choice: dict):
    delta = choice.get("delta") or {}
    if "delta" in choice:
        if "reasoning_content" in delta:
            reasoning = delta.get("reasoning_content")
        elif "reasoning" in delta:
            reasoning = delta.get("reasoning")
        else:
            reasoning = None
        return delta.get("content") or "", reasoning, True
    return choice.get("text", ""), None, False


def _tt_format_sse_response(
    accumulated_text: dict,
    accumulated_reasoning: dict,
    uses_chat_chunks: bool,
) -> dict:
    choices = []
    indexes = sorted(set(accumulated_text) | set(accumulated_reasoning))
    for index in indexes:
        text = accumulated_text.get(index, "")
        if uses_chat_chunks:
            message = {"content": text}
            if index in accumulated_reasoning:
                message["reasoning_content"] = accumulated_reasoning[index]
            choices.append({"index": index, "message": message})
        else:
            choices.append({"index": index, "text": text})
    return _tt_wrap_chat_response_reasoning({"choices": choices})


async def _tt_consume_sse_stream(self, response) -> dict:
    accumulated_text = {}
    accumulated_reasoning = {}
    uses_chat_chunks = False
    try:
        while True:
            line_bytes = await response.content.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                index = choice.get("index", 0)
                text, reasoning, is_chat_chunk = _tt_stream_chunk_parts(choice)
                uses_chat_chunks = uses_chat_chunks or is_chat_chunk
                accumulated_text[index] = accumulated_text.get(index, "") + text
                if reasoning is not None:
                    accumulated_reasoning[index] = (
                        accumulated_reasoning.get(index, "") + reasoning
                    )
    except BaseException as exc:
        if not accumulated_text and not accumulated_reasoning:
            raise
        prefix = "__PARTIAL_OUTPUT__ (" + repr(exc) + "): "
        indexes = set(accumulated_text) | set(accumulated_reasoning)
        accumulated_text = {
            i: prefix + accumulated_text.get(i, "") for i in indexes
        }
    return _tt_format_sse_response(
        accumulated_text,
        accumulated_reasoning,
        uses_chat_chunks,
    )


def _tt_consume_requests_sse_stream(response) -> dict:
    accumulated_text = {}
    accumulated_reasoning = {}
    uses_chat_chunks = False
    try:
        for line in response.iter_lines(decode_unicode=True):
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            line = str(line or "").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            for choice in chunk.get("choices", []):
                index = choice.get("index", 0)
                text, reasoning, is_chat_chunk = _tt_stream_chunk_parts(choice)
                uses_chat_chunks = uses_chat_chunks or is_chat_chunk
                accumulated_text[index] = accumulated_text.get(index, "") + text
                if reasoning is not None:
                    accumulated_reasoning[index] = (
                        accumulated_reasoning.get(index, "") + reasoning
                    )
    except BaseException as exc:
        if not accumulated_text and not accumulated_reasoning:
            raise
        prefix = "__PARTIAL_OUTPUT__ (" + repr(exc) + "): "
        indexes = set(accumulated_text) | set(accumulated_reasoning)
        accumulated_text = {
            i: prefix + accumulated_text.get(i, "") for i in indexes
        }
    return _tt_format_sse_response(
        accumulated_text,
        accumulated_reasoning,
        uses_chat_chunks,
    )


def _tt_model_call(self, messages, *, generate: bool = True, gen_kwargs: dict = None, **kwargs):
    gen_kwargs = copy.deepcopy(gen_kwargs)
    payload = self._create_payload(
        self.create_message(messages),
        generate=generate,
        gen_kwargs=gen_kwargs,
        seed=self._seed,
        eos=self.eos_string,
        **kwargs,
    )
    is_streaming = generate and str(payload.get("stream", False)).lower() == "true"
    try:
        response = requests.post(
            self.base_url,
            json=payload,
            headers=self.header,
            verify=self.verify_certificate,
            stream=is_streaming,
        )
        if not response.ok:
            eval_logger.warning(
                "API request failed with error message: " + response.text + ". Retrying..."
            )
        response.raise_for_status()
        if is_streaming:
            return _tt_consume_requests_sse_stream(response)
        return _tt_wrap_chat_response_reasoning(response.json())
    except RetryError:
        eval_logger.error(
            "API request failed after multiple retries. Please check the API status."
        )
        return None


TemplateAPI._consume_sse_stream = _tt_consume_sse_stream
TemplateAPI.model_call = _tt_model_call
# === end TT patch ===
"""

_LM_EVAL_REASONING_LOG_PATCH = """

# === TT patch: preserve chat reasoning in sample logs ===
# Applied post-install by workflows.workflow_venvs.patch_evals_common_chat_streaming.
def _tt_extract_reasoning(value):
    if isinstance(value, (list, tuple)):
        found = False
        extracted = []
        for item in value:
            item_found, item_reasoning = _tt_extract_reasoning(item)
            found = found or item_found
            extracted.append(item_reasoning)
        return found, extracted
    if hasattr(value, "reasoning_content"):
        return True, value.reasoning_content
    return False, None


_tt_original_save_results_samples = EvaluationTracker.save_results_samples


def _tt_save_results_samples(self, task_name: str, samples: dict) -> None:
    for sample in samples:
        found, reasoning = _tt_extract_reasoning(sample.get("resps", []))
        if found:
            sample["reasoning_content"] = reasoning
    return _tt_original_save_results_samples(self, task_name, samples)


EvaluationTracker.save_results_samples = _tt_save_results_samples
# === end TT patch ===
"""


# Sentinel marker appended to lm_eval/models/openai_completions.py so the
# reasoning-effort patch is applied at most once per venv (idempotent across
# repeated setup runs).
_LM_EVAL_REASONING_EFFORT_SENTINEL = (
    "# === TT patch: gpt-oss reasoning_effort -> chat_template_kwargs ==="
)

# Monkeypatch appended to the END of lm_eval/models/openai_completions.py (after
# LocalChatCompletion is defined). The eval sends `reasoning_effort` as a
# top-level chat-completions field. vLLM applies it directly (build_chat_params
# folds it into chat_template_kwargs, and the gpt-oss Harmony path reads
# request.reasoning_effort). The NVIDIA-Dynamo frontend used by the blaze /
# tt-media-server deployment does NOT: it validates `reasoning_effort` but only
# forwards `chat_template_kwargs` into the chat-template render context, so
# gpt-oss renders at the template's default effort ("medium") regardless of what
# is requested. Mirror vLLM's build_chat_params principle by folding the
# top-level field into `chat_template_kwargs` at payload-build time. The
# top-level value is preserved so the vLLM path is unchanged; each backend uses
# whichever it understands.
_LM_EVAL_REASONING_EFFORT_PATCH = """

# === TT patch: gpt-oss reasoning_effort -> chat_template_kwargs ===
# Applied post-install by workflows.workflow_venvs.patch_evals_common_chat_streaming.
_tt_orig_create_payload = LocalChatCompletion._create_payload


def _tt_create_payload_with_reasoning_effort(self, *args, **kwargs):
    payload = _tt_orig_create_payload(self, *args, **kwargs)
    effort = payload.get("reasoning_effort")
    if effort is not None:
        cta = payload.setdefault("chat_template_kwargs", {})
        if isinstance(cta, dict):
            cta.setdefault("reasoning_effort", effort)
    return payload


LocalChatCompletion._create_payload = _tt_create_payload_with_reasoning_effort
# === end TT patch ===
"""


def patch_evals_common_chat_streaming(
    venv_config: VenvConfig,
    model_spec: "ModelSpec",
) -> bool:
    """Hook for EVALS_COMMON: fix lm-eval chat-completions SSE streaming.

    lm-eval's streaming path was written for the text-completions API and breaks
    chat-completions streaming (KeyError: 'message'). Streaming is required
    against the remote console to avoid 504s on long reasoning generations, so we
    patch the installed package in place rather than disabling streaming. The
    patch is appended once (guarded by a sentinel) to the module so it survives as
    long as the venv exists; venv rebuilds re-apply it.
    """
    matches = sorted(
        venv_config.venv_path.glob(
            "lib/python*/site-packages/lm_eval/models/api_models.py"
        )
    )
    if not matches:
        logger.warning(
            "Could not locate lm_eval/models/api_models.py under "
            f"{venv_config.venv_path}; chat-completions streaming patch not "
            "applied. Streaming evals may fail to parse generations."
        )
        return True
    api_models_path = matches[0]
    api_models_text = api_models_path.read_text()
    if _LM_EVAL_CHAT_STREAM_SENTINEL in api_models_text:
        logger.info(f"chat-streaming patch already present in {api_models_path}")
    else:
        api_models_path.write_text(api_models_text + _LM_EVAL_CHAT_STREAM_PATCH)
        logger.info(f"applied chat-streaming patch to {api_models_path}")

    # Route gpt-oss reasoning_effort into chat_template_kwargs so the NVIDIA-Dynamo
    # frontend (blaze / tt-media-server deployment) renders the requested reasoning
    # effort instead of the chat-template default. See _LM_EVAL_REASONING_EFFORT_PATCH.
    oai_path = api_models_path.parent / "openai_completions.py"
    if not oai_path.is_file():
        logger.warning(
            f"Could not locate {oai_path}; gpt-oss reasoning_effort will not be "
            "routed into chat_template_kwargs (the Dynamo/blaze path renders at "
            "the chat-template default reasoning effort)."
        )
    else:
        oai_text = oai_path.read_text()
        if _LM_EVAL_REASONING_EFFORT_SENTINEL in oai_text:
            logger.info(f"reasoning-effort patch already present in {oai_path}")
        else:
            oai_path.write_text(oai_text + _LM_EVAL_REASONING_EFFORT_PATCH)
            logger.info(f"applied reasoning-effort patch to {oai_path}")

    tracker_path = api_models_path.parents[1] / "loggers" / "evaluation_tracker.py"
    if not tracker_path.is_file():
        logger.warning(
            f"Could not locate {tracker_path}; reasoning content will not be "
            "included in lm-eval sample logs."
        )
        return True
    tracker_text = tracker_path.read_text()
    if _LM_EVAL_REASONING_LOG_SENTINEL in tracker_text:
        logger.info(f"reasoning-log patch already present in {tracker_path}")
    else:
        tracker_path.write_text(tracker_text + _LM_EVAL_REASONING_LOG_PATCH)
        logger.info(f"applied reasoning-log patch to {tracker_path}")
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
        setup_function=patch_evals_common_chat_streaming,
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
    VenvConfig(
        venv_type=WorkflowVenvType.V2_RUN_SCRIPT,
        requirements_file="v2-run-script.txt",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.V2_PREFIX_CACHE,
        requirements_file="v2-prefix-cache.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.V2_LLM_VLLM,
        requirements_file="v2-llm-vllm.txt",
        # Force transformers 5.x past vllm==0.13.0's `transformers<5` cap so the
        # gemma-4 tokenizer loads; keeps vllm (and the bench-serve client) at
        # 0.13.0 for every other model. See v2-llm-vllm-overrides.txt.
        overrides_file="v2-llm-vllm-overrides.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.V2_LLM_GUIDELLM,
        requirements_file="v2-llm-guidellm.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.V2_LLM_AIPERF,
        requirements_file="v2-llm-aiperf.txt",
        extra_dirs=("artifacts",),
        python_version="3.11",
    ),
    VenvConfig(
        venv_type=WorkflowVenvType.V2_SPEC_DECODE,
        requirements_file="v2-spec-decode.txt",
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
