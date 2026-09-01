# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import hashlib
import logging
import os
import re
import stat
from pathlib import Path, PurePosixPath

from llm_module.eval_configs import (
    filter_agentic_tasks_by_benchmark,
    filter_tasks_by_min_context,
)
from reference_config.benchmarking.benchmark_config import get_benchmark_config
from reference_config.evals.eval_config import EVAL_CONFIGS
from workflows.model_spec import MODEL_SPECS
from workflows.utils import (
    MIN_SUPPORTED_IMAGE_VERSION,
    check_path_permissions_for_uid,
    ensure_readwriteable_dir,
    get_default_workflow_root_log_dir,
    get_groups_for_uid,
    get_repo_root_path,
    parse_version_tuple,
    resolve_hf_snapshot_dir,
    run_command,
)
from workflows.workflow_dispatch import can_dispatch_to_engine
from workflows.workflow_types import (
    DeviceTypes,
    EvalLimitMode,
    InferenceEngine,
    WorkflowType,
    WorkflowVenvType,
)
from workflows.workflow_venvs import VENV_CONFIGS

logger = logging.getLogger("run_log")

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_REVISION_RE = re.compile(r"[0-9a-f]{40}")
_PACKAGE_ID_RE = re.compile(r"sha256(?:-v[0-9]+)?(?:-[0-9a-f]{64}){2,3}")
_QUETZAL_PACKAGE_PARENT = PurePosixPath("/home/container_app_user/quetzal/packages")


def validate_quetzal_models_ci_contract(model_spec, runtime_config) -> None:
    """Reject incomplete generated-only release enrollments before setup."""
    if _agentic_impl_id(model_spec) != "quetzal":
        return
    if WorkflowType.from_string(runtime_config.workflow) != WorkflowType.RELEASE:
        return

    env = model_spec.env_vars
    required = {
        "QUETZAL_PACKAGE_ID",
        "QUETZAL_PACKAGE_ROOT",
        "QZ_MODELS_ROOT",
        "QZ_QUALIFICATION_MANIFEST",
        "QUETZAL_BUNDLE_MANIFEST_SHA256",
        "QUETZAL_REQUIRED_SOURCE_REVISION",
        "QUETZAL_REQUIRED_TT_METAL_COMMIT",
        "QUETZAL_TT_METAL_PATCHSET_STATUS",
        "QUETZAL_PREFILL_GENERATED_PY",
        "QUETZAL_PREFILL_METADATA_JSON",
        "QUETZAL_DECODE_GENERATED_PY",
        "QUETZAL_DECODE_METADATA_JSON",
        "QUETZAL_WEIGHTS",
    }
    missing = sorted(name for name in required if not env.get(name))
    if missing:
        raise ValueError(
            "Quetzal Models-CI enrollment is incomplete: missing "
            f"{missing}; release rows require generated package, source, "
            "TT-Metal, and explicit patchset-status identities"
        )
    if env.get("VLLM_PLUGINS") != "quetzal_model_registry,tt":
        raise ValueError(
            "Quetzal Models-CI enrollment requires "
            "VLLM_PLUGINS=quetzal_model_registry,tt"
        )
    if str(env.get("TT_VLLM_BUILTIN_MODELS")) != "0":
        raise ValueError(
            "Quetzal Models-CI enrollment requires TT_VLLM_BUILTIN_MODELS=0"
        )
    if str(env.get("QUETZAL_VLLM")) != "1":
        raise ValueError("Quetzal Models-CI enrollment requires QUETZAL_VLLM=1")

    package_id = str(env["QUETZAL_PACKAGE_ID"])
    if _PACKAGE_ID_RE.fullmatch(package_id) is None:
        raise ValueError("Quetzal Models-CI package ID must be content-addressed")
    package_root = PurePosixPath(str(env["QUETZAL_PACKAGE_ROOT"]))
    expected_root = _QUETZAL_PACKAGE_PARENT / package_id
    if (
        package_root != expected_root
        or PurePosixPath(str(env["QZ_MODELS_ROOT"])) != expected_root
    ):
        raise ValueError(
            "Quetzal Models-CI package roots must name the exact content ID"
        )
    if PurePosixPath(str(env["QZ_QUALIFICATION_MANIFEST"])) != (
        expected_root / "qualification_manifest.yaml"
    ):
        raise ValueError(
            "Quetzal Models-CI qualification manifest must be inside the exact package"
        )
    for key in (
        "QUETZAL_PREFILL_GENERATED_PY",
        "QUETZAL_PREFILL_METADATA_JSON",
        "QUETZAL_DECODE_GENERATED_PY",
        "QUETZAL_DECODE_METADATA_JSON",
        "QUETZAL_WEIGHTS",
    ):
        if not PurePosixPath(str(env[key])).is_relative_to(expected_root):
            raise ValueError(f"{key} must remain inside the exact Quetzal package")
    for key in ("QUETZAL_BUNDLE_MANIFEST_SHA256",):
        if _SHA256_RE.fullmatch(str(env[key])) is None:
            raise ValueError(f"{key} must be an exact lowercase SHA-256")
    attestation_sha = env.get("QUETZAL_RUNTIME_ATTESTATION_SHA256")
    if attestation_sha is None:
        logger.warning(
            "Quetzal Models-CI publication provenance warning [unattested]: "
            "no runtime compatibility attestation is catalogued; functional "
            "and quality qualification remains runnable"
        )
    elif _SHA256_RE.fullmatch(str(attestation_sha)) is None:
        raise ValueError(
            "Quetzal Models-CI functional blocker: "
            "QUETZAL_RUNTIME_ATTESTATION_SHA256 must be an exact lowercase SHA-256"
        )
    if _GIT_REVISION_RE.fullmatch(str(env["QUETZAL_REQUIRED_SOURCE_REVISION"])) is None:
        raise ValueError(
            "QUETZAL_REQUIRED_SOURCE_REVISION must be an exact lowercase git commit"
        )
    if _GIT_REVISION_RE.fullmatch(str(env["QUETZAL_REQUIRED_TT_METAL_COMMIT"])) is None:
        raise ValueError(
            "QUETZAL_REQUIRED_TT_METAL_COMMIT must be an exact lowercase git commit"
        )
    patchset_status = env["QUETZAL_TT_METAL_PATCHSET_STATUS"]
    patchset_sha = env.get("QUETZAL_REQUIRED_TT_METAL_PATCHSET_SHA256")
    if patchset_status == "applied":
        if _SHA256_RE.fullmatch(str(patchset_sha or "")) is None:
            raise ValueError(
                "applied TT-Metal patchset requires an exact patchset SHA-256"
            )
    elif patchset_status == "none":
        if patchset_sha:
            raise ValueError(
                "TT-Metal patchset status 'none' cannot carry a patchset SHA-256"
            )
    else:
        raise ValueError("QUETZAL_TT_METAL_PATCHSET_STATUS must be 'applied' or 'none'")

    selected = _selected_agentic_tasks(model_spec, runtime_config)
    swe_tasks = [task for task in selected if task.task_name == "swe_bench_verified"]
    if len(swe_tasks) != 1:
        raise ValueError(
            "Quetzal Models-CI release requires exactly one context-admitted "
            "SWE-bench Verified task with a predeclared graded evaluation contract"
        )
    task = swe_tasks[0]
    cfg = task.swebench_eval_config
    if cfg is None or cfg.agent_backend != "mini-swe-agent":
        raise ValueError(
            "Quetzal Models-CI SWE enrollment must use the common "
            "mini-swe-agent verifier path"
        )
    required_context, max_input, max_output, _ = _agentic_context_requirement(task)
    available_context = getattr(model_spec.device_model_spec, "max_context", None)
    if not isinstance(available_context, int) or required_context > available_context:
        raise ValueError(
            "Quetzal Models-CI SWE envelope exceeds the admitted serving profile: "
            f"required={required_context}, available={available_context}, "
            f"input={max_input}, output={max_output}"
        )
    if (
        not isinstance(cfg.mini_observation_chars, int)
        or isinstance(cfg.mini_observation_chars, bool)
        or cfg.mini_observation_chars < 2
    ):
        raise ValueError(
            "Quetzal Models-CI SWE enrollment requires an explicit positive "
            "mini_observation_chars payload-retention cap"
        )
    step_limit = cfg.mini_agent_kwargs.get("step_limit")
    if (
        not isinstance(step_limit, int)
        or isinstance(step_limit, bool)
        or step_limit <= 0
    ):
        raise ValueError(
            "Quetzal Models-CI SWE enrollment requires an explicit positive step_limit"
        )
    if (
        not isinstance(cfg.instance_selection_provenance, str)
        or not cfg.instance_selection_provenance.strip()
    ):
        raise ValueError(
            "Quetzal Models-CI SWE enrollment requires reviewed, "
            "model-output-independent instance selection provenance"
        )
    if cfg.qualification_claim != "models_ci_graded":
        raise ValueError(
            "Quetzal Models-CI release cannot use a local/report-only SWE claim; "
            "qualification_claim must be 'models_ci_graded'"
        )
    score = task.score
    mode_references = getattr(score, "mode_reference_scores", {}) if score else {}
    smoke_ids = cfg.instance_ids_map.get(EvalLimitMode.SMOKE_TEST)
    if (
        not isinstance(smoke_ids, list)
        or not smoke_ids
        or not all(
            isinstance(instance_id, str) and instance_id for instance_id in smoke_ids
        )
    ):
        raise ValueError(
            "Quetzal Models-CI SWE enrollment requires a predeclared nonempty "
            "smoke diagnostic set independent of model output"
        )
    if cfg.selection_policy == "reviewed_fixed_subset":
        nightly_ids = cfg.instance_ids_map.get(EvalLimitMode.CI_NIGHTLY)
        if (
            not isinstance(nightly_ids, list)
            or not nightly_ids
            or not all(
                isinstance(instance_id, str) and instance_id
                for instance_id in nightly_ids
            )
        ):
            raise ValueError(
                "reviewed_fixed_subset requires a predeclared nonempty "
                "CI_NIGHTLY instance set"
            )
        if EvalLimitMode.CI_NIGHTLY not in mode_references:
            raise ValueError(
                "reviewed_fixed_subset requires an independently baselined "
                "CI_NIGHTLY mode reference for the exact fixed SWE subset"
            )
        if _GIT_REVISION_RE.fullmatch(str(cfg.dataset_revision or "")) is None:
            raise ValueError(
                "reviewed_fixed_subset requires an exact pinned dataset revision"
            )
        if len(set(nightly_ids)) != len(nightly_ids):
            raise ValueError(
                "reviewed_fixed_subset CI_NIGHTLY instance IDs must be unique"
            )
        expected_subset_digest = hashlib.sha256(
            json.dumps(nightly_ids, ensure_ascii=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if cfg.ordered_instance_ids_sha256 != expected_subset_digest:
            raise ValueError(
                "reviewed_fixed_subset requires the exact SHA-256 of its "
                "canonical ordered CI_NIGHTLY instance IDs"
            )
        subset_reference = mode_references[EvalLimitMode.CI_NIGHTLY]
        if (
            not isinstance(subset_reference.score, (int, float))
            or isinstance(subset_reference.score, bool)
            or subset_reference.score <= 0
            or not isinstance(subset_reference.ref, str)
            or not subset_reference.ref.strip()
        ):
            raise ValueError(
                "reviewed_fixed_subset requires a positive measured score and "
                "nonempty reference tied to the exact CI_NIGHTLY subset"
            )
    elif cfg.selection_policy == "full_dataset":
        raise ValueError(
            "full_dataset release qualification is disabled until the contract "
            "binds authoritative dataset cardinality and an exact full ID-set "
            "digest, then verifies complete report coverage"
        )
    else:
        raise ValueError(
            "Quetzal Models-CI SWE enrollment requires selection_policy="
            "'reviewed_fixed_subset' or 'full_dataset'"
        )


def _agentic_impl_id(model_spec) -> str:
    impl = getattr(model_spec, "impl", None)
    return str(getattr(impl, "impl_id", impl) or "unknown")


def _external_agentic_task_name(model_spec, runtime_config) -> str | None:
    """Return the task bound by an external launch contract, if configured.

    The full cryptographic/live identity check remains in the agentic runner.
    This small preflight read exists only to make capability admission select
    the same one task before host setup or a device-backed server is started.
    """
    contract_path = getattr(runtime_config, "external_agentic_contract", None)
    if not contract_path:
        return None
    try:
        document = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"cannot read external agentic contract {contract_path}: {exc}"
        ) from exc
    contract = document.get("contract") if isinstance(document, dict) else None
    if not isinstance(contract, dict):
        raise RuntimeError("external agentic contract has no contract object")
    expected_repo = model_spec.hf_model_repo
    if contract.get("hf_model_repo") != expected_repo:
        raise RuntimeError(
            "external agentic contract model differs from runtime model: "
            f"expected {expected_repo!r}, got {contract.get('hf_model_repo')!r}"
        )
    task_name = contract.get("task")
    if not isinstance(task_name, str) or not task_name:
        raise RuntimeError("external agentic contract needs non-empty task")
    return task_name


def _selected_agentic_tasks(model_spec, runtime_config) -> list:
    eval_config = EVAL_CONFIGS.get(model_spec.model_name)
    if eval_config is None:
        return []
    tasks = [
        task
        for task in getattr(eval_config, "tasks", [])
        if task.workflow_venv_type == WorkflowVenvType.EVALS_AGENTIC
    ]
    selection = getattr(runtime_config, "agentic_benchmark", None)
    if isinstance(selection, str) and selection.strip():
        tasks = filter_agentic_tasks_by_benchmark(tasks, selection)

    exact_task = _external_agentic_task_name(model_spec, runtime_config)
    if exact_task is not None:
        selected = [task for task in tasks if task.task_name == exact_task]
        if len(selected) != 1:
            raise RuntimeError(
                "external agentic contract must select exactly one configured "
                f"agentic task {exact_task!r}; selected "
                f"{[task.task_name for task in selected]!r}"
            )
        return selected

    # A default release may omit a task only when the catalogue explicitly
    # declares a context floor. An explicit --agentic-benchmark selection and
    # standalone agentic runs retain the task and fail capability admission,
    # rather than silently turning a requested evaluation into a no-op.
    if WorkflowType.from_string(
        runtime_config.workflow
    ) == WorkflowType.RELEASE and not (
        isinstance(selection, str) and selection.strip()
    ):
        tasks = filter_tasks_by_min_context(tasks, model_spec)
    return tasks


def _positive_token_limit(value, *, field: str, task_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            "Agentic capability admission failed: "
            f"task={task_name!r} required_context=undeclared; "
            f"{field} must be a positive integer, got {value!r}"
        )
    return value


def _agentic_context_requirement(task) -> tuple[int, int, int, int | None]:
    """Return the effective context floor and its declared components.

    ``max_input_tokens + max_output_tokens`` is the request payload.  A task's
    optional ``min_context_required`` can reserve additional chat-template,
    tool-definition, or harness headroom, and therefore takes precedence when
    it is larger than the raw payload.
    """
    if task.agentic_eval_config is not None:
        agent_kwargs = task.agentic_eval_config.agent_kwargs
        model_info = (
            agent_kwargs.get("model_info") if isinstance(agent_kwargs, dict) else None
        )
        if not isinstance(model_info, dict):
            raise ValueError(
                "Agentic capability admission failed: "
                f"task={task.task_name!r} required_context=undeclared; "
                "agentic_eval_config.agent_kwargs.model_info must be an object "
                "declaring max_input_tokens and max_output_tokens"
            )
        max_input = model_info.get("max_input_tokens")
        max_output = model_info.get("max_output_tokens")
        source = "agentic_eval_config.agent_kwargs.model_info"
    elif task.swebench_eval_config is not None:
        max_input = task.swebench_eval_config.max_input_tokens
        max_output = task.swebench_eval_config.max_output_tokens
        source = "swebench_eval_config"
    else:
        raise ValueError(
            "Agentic capability admission failed: "
            f"task={task.task_name!r} required_context=undeclared; "
            "selected EVALS_AGENTIC task has no agentic or SWE-bench config"
        )

    max_input = _positive_token_limit(
        max_input,
        field=f"{source}.max_input_tokens",
        task_name=task.task_name,
    )
    max_output = _positive_token_limit(
        max_output,
        field=f"{source}.max_output_tokens",
        task_name=task.task_name,
    )
    minimum = getattr(task, "min_context_required", None)
    if minimum is not None and (
        not isinstance(minimum, int) or isinstance(minimum, bool) or minimum <= 0
    ):
        raise ValueError(
            "Agentic capability admission failed: "
            f"task={task.task_name!r} required_context=undeclared; "
            f"min_context_required must be a positive integer, got {minimum!r}"
        )
    required = max(max_input + max_output, minimum or 0)
    return required, max_input, max_output, minimum


def validate_agentic_task_capabilities(model_spec, runtime_config) -> None:
    """Fail before host/server/device setup if an agentic task cannot fit.

    The task's input and output budgets are an end-to-end request envelope, so
    their sum must fit the exact selected DeviceModelSpec context.  Harness
    concurrency is deliberately not compared with max_concurrency: clients may
    queue requests and that policy is independent from per-request context.
    """
    workflow_type = WorkflowType.from_string(runtime_config.workflow)
    if workflow_type not in (WorkflowType.AGENTIC, WorkflowType.RELEASE):
        return

    tasks = _selected_agentic_tasks(model_spec, runtime_config)
    if not tasks:
        return

    device_spec = getattr(model_spec, "device_model_spec", None)
    available = getattr(device_spec, "max_context", None)
    if not isinstance(available, int) or isinstance(available, bool) or available <= 0:
        raise ValueError(
            "Agentic capability admission failed: "
            f"model={model_spec.model_name!r} "
            f"implementation={_agentic_impl_id(model_spec)!r} "
            "available_context=undeclared; selected DeviceModelSpec.max_context "
            f"must be a positive integer, got {available!r}"
        )

    for task in tasks:
        try:
            required, max_input, max_output, minimum = _agentic_context_requirement(
                task
            )
        except ValueError as exc:
            raise ValueError(
                f"model={model_spec.model_name!r} "
                f"implementation={_agentic_impl_id(model_spec)!r} "
                f"device={getattr(device_spec, 'device', 'unknown')!r} "
                f"available_context={available}; {exc}"
            ) from exc
        if required > available:
            minimum_detail = (
                f"; min_context_required={minimum}" if minimum is not None else ""
            )
            raise ValueError(
                "Agentic capability admission failed before host/server/device "
                f"setup: model={model_spec.model_name!r} "
                f"implementation={_agentic_impl_id(model_spec)!r} "
                f"device={getattr(device_spec, 'device', 'unknown')!r} "
                f"task={task.task_name!r} available_context={available} "
                f"required_context={required} "
                f"(max_input_tokens={max_input} + max_output_tokens={max_output}"
                f"{minimum_detail})"
            )


def _uses_external_runtime_model_spec(runtime_config) -> bool:
    return bool(runtime_config.runtime_model_spec_json)


def _swarmone_license_available() -> bool:
    """Whether a SwarmOne swo-bench license can be resolved.

    Mirrors swo-bench's own resolution order (env var, then the config file);
    the key itself is never read into run.py, only its presence is checked.
    """
    if os.environ.get("SWO_LICENSE_KEY"):
        return True
    key_file = Path.home() / ".swarmone" / "license.key"
    try:
        return key_file.is_file() and bool(key_file.read_text().strip())
    except OSError:
        return False


def _check_image_version_supported(model_spec):
    """Refuse to run a pre-0.11 vLLM image with this run.py.

    The vLLM docker image interface was reshaped in v0.11.0 (commit 50db8ac7
    "Simplify and improve vLLM Docker image interface"): ENTRYPOINT changed
    from docker-entrypoint.sh + gosu to bash -c, the script's CLI argument
    contract changed, and shared-memory + env-var conventions changed. main
    only emits the new contract, so an older vLLM image won't start.

    Scoped to vLLM only — media-inference-server and forge images have
    different Dockerfiles and aren't affected by this interface change
    (the docker command for them is also simpler and stable across versions).

    apply_overrides re-parses model_spec.version from --override-docker-image
    when present, so this check covers both template-pinned versions and
    override paths.
    """
    if model_spec.inference_engine != InferenceEngine.VLLM.value:
        return
    parsed = parse_version_tuple(model_spec.version)
    if parsed is None:
        # Unparseable versions (`dev`, `latest`, etc.) default to "newest
        # contract" — let the runtime decide, matches main's behaviour.
        return
    if parsed < MIN_SUPPORTED_IMAGE_VERSION:
        min_str = ".".join(str(p) for p in MIN_SUPPORTED_IMAGE_VERSION)
        tag = f"v{model_spec.version}"
        raise RuntimeError(
            f"⛔ Image v{model_spec.version} is not supported in this "
            f"version of run.py (need v{min_str}+). Check out the matching "
            f"release tag {tag} and re-run:\n"
            f"    git checkout {tag}"
        )


def validate_runtime_args(model_spec, runtime_config):
    args = runtime_config
    workflow_type = WorkflowType.from_string(args.workflow)

    if not args.device:
        # TODO: detect phy device
        raise NotImplementedError("Device detection not implemented yet")

    model_id = model_spec.model_id

    # Catalog runs must resolve to MODEL_SPECS; --runtime-model-spec-json and
    # --custom-weights supply their own spec whose model_id is not in the catalog.
    if (
        model_id not in MODEL_SPECS
        and not _uses_external_runtime_model_spec(args)
        and not getattr(args, "custom_weights", None)
    ):
        raise ValueError(
            f"model:={runtime_config.model} does not support device:={runtime_config.device}"
        )

    # The exact ModelSpec and task selection are now known, while no host
    # storage, server process, model payload, or device has been touched yet.
    validate_quetzal_models_ci_contract(model_spec, runtime_config)
    validate_agentic_task_capabilities(model_spec, runtime_config)

    # The image-version contract only matters when run.py actually launches the
    # vLLM docker image. Client-side / external-server runs (no --docker-server)
    # — including the v2-routed prefill_decode / prefix-cache / spec-decode
    # workflows that bring up or target their own server — never emit a docker
    # command, so the pinned image version is irrelevant and must not gate them.
    if args.docker_server:
        _check_image_version_supported(model_spec)

    assert not (args.docker_server and args.local_server), (
        "Cannot run --docker-server and --local-server"
    )

    if workflow_type == WorkflowType.EVALS:
        assert model_spec.model_name in EVAL_CONFIGS, (
            f"Model:={model_spec.model_name} not found in EVAL_CONFIGS"
        )
    if (
        workflow_type == WorkflowType.BENCHMARKS
        and not getattr(args, "prefix_cache", False)
        and not getattr(args, "spec_decode", False)
        and not can_dispatch_to_engine(model_spec, runtime_config)
    ):
        if os.getenv("OVERRIDE_BENCHMARKS"):
            logger.warning("OVERRIDE_BENCHMARKS is active, using override benchmarks")
        get_benchmark_config(model_spec)
    if workflow_type == WorkflowType.AGENTIC_TRACES or (
        workflow_type == WorkflowType.RELEASE and getattr(args, "agentic_traces", False)
    ):
        # Fail here rather than after the multi-minute InferenceX clone + install
        # that the AGENTIC_TRACES venv setup performs -- and, for a release run,
        # rather than after the evals and benchmarks that precede the child.
        from reference_config.agentic_traces.agentic_traces_config import (
            TraceSource,
            default_run_specs,
            get_agentic_traces_config,
        )

        agentic_traces_config = get_agentic_traces_config(model_spec)
        assert agentic_traces_config is not None, (
            f"Model:={model_spec.model_name} (model_id={model_spec.model_id}) has "
            "no AGENTIC_TRACES_CONFIGS entry. Add one to "
            "reference_config/agentic_traces/agentic_traces_config.py, including "
            "the InferenceX git ref to pin."
        )

        # A SwarmOne run needs a swo-bench license; require it up front (like
        # HF_TOKEN) rather than failing minutes into the run inside the driver.
        # Only when SwarmOne will actually run, though: it is an opt-in source,
        # so a model that merely has a SwarmOne run configured still runs its
        # plain sweep (InferenceX only) without a license.
        sources_arg = getattr(args, "agentic_traces_sources", None)
        if sources_arg:
            selected = {
                part.strip().lower().replace("-", "_")
                for part in sources_arg.split(",")
                if part.strip()
            }
            swarmone_will_run = TraceSource.SWARMONE.value in selected
        else:
            swarmone_will_run = any(
                run.trace_source is TraceSource.SWARMONE
                for run in default_run_specs(agentic_traces_config)
            )
        if swarmone_will_run and not _swarmone_license_available():
            raise ValueError(
                "⛔ The swarmone agentic-traces source requires a SwarmOne "
                "license. Set the SWO_LICENSE_KEY environment variable or write "
                "the key to ~/.swarmone/license.key. Request a key from "
                "benb@swarmone.ai. To run without SwarmOne, drop "
                "`--agentic-traces-sources swarmone`."
            )

    if workflow_type == WorkflowType.STRESS_TESTS:
        pass  # Model support already validated via MODEL_SPECS check

    if workflow_type == WorkflowType.SERVER:
        if not (args.docker_server or args.local_server):
            raise ValueError(
                f"Workflow {args.workflow} requires --docker-server or --local-server"
            )
        if (
            args.local_server
            and model_spec.inference_engine != InferenceEngine.VLLM.value
        ):
            raise NotImplementedError(
                "--local-server currently supports only vLLM-backed model specs"
            )

        # For partitioning Galaxy per tray as T3K
        # TODO: Add a check to verify whether these devices belong to the same tray
        if DeviceTypes.from_string(args.device) == DeviceTypes.GALAXY_T3K:
            if not args.device_id or len(args.device_id) != 8:
                raise ValueError(
                    "Galaxy T3K requires exactly 8 device IDs specified with --device-id (e.g. '0,1,2,3,4,5,6,7'). These must be devices within the same tray."
                )

    if workflow_type == WorkflowType.RELEASE:
        # NOTE: fail fast for models without both defined evals and generated
        # benchmark tasks. A run_*.log file will be made for failed combinations.
        assert model_spec.model_name in EVAL_CONFIGS, (
            f"Model:={model_spec.model_name} not found in EVAL_CONFIGS"
        )
        if not can_dispatch_to_engine(model_spec, runtime_config):
            get_benchmark_config(model_spec)

    if DeviceTypes.from_string(args.device) == DeviceTypes.GPU:
        if args.docker_server or args.local_server:
            raise NotImplementedError(
                "GPU support for running inference server not implemented yet"
            )

    if args.local_server and not args.tt_metal_home:
        raise ValueError(
            "--local-server requires --tt-metal-home or TT_METAL_HOME to be set"
        )

    # Validate mutual exclusivity of weight source options
    weight_source_args = [
        args.host_volume,
        args.host_hf_cache,
        getattr(args, "host_weights_dir", None),
    ]
    if sum(1 for a in weight_source_args if a) > 1:
        raise ValueError(
            "Only one of --host-volume, --host-hf-cache, --host-weights-dir can be specified."
        )

    if "ENABLE_AUTO_TOOL_CHOICE" in os.environ:
        raise AssertionError(
            "Setting ENABLE_AUTO_TOOL_CHOICE has been deprecated, use the VLLM_OVERRIDE_ARGS env var directly or via --vllm-override-args in run.py CLI.\n"
            'Enable auto tool choice by adding --vllm-override-args \'{"enable-auto-tool-choice": true, "tool-call-parser": <parser-name>}\' when calling run.py'
        )


def _get_local_server_python_env_dir(runtime_config) -> Path:
    tt_metal_home = Path(runtime_config.tt_metal_home).expanduser().resolve()
    if runtime_config.tt_metal_python_venv_dir:
        return Path(runtime_config.tt_metal_python_venv_dir).expanduser().resolve()
    return tt_metal_home / "python_env"


def _validate_local_vllm_installation(runtime_config):
    venv_python = _get_local_server_python_env_dir(runtime_config) / "bin" / "python"
    if not venv_python.exists():
        raise ValueError(f"⛔ Missing required python venv interpreter: {venv_python}")

    return_code = run_command([str(venv_python), "-c", "import vllm"], logger=logger)
    if return_code != 0:
        raise ValueError(
            "⛔ --local-server with inference engine vLLM requires the `vllm` Python "
            f"package to be installed in the tt-metal python environment. Could not "
            f"import `vllm` with: {venv_python}"
        )
    logger.info(f"✅ validated vLLM Python package import with: {venv_python}")

    _validate_local_vllm_tt_plugin(runtime_config, venv_python)


def _validate_local_vllm_tt_plugin(runtime_config, venv_python: Path):
    """Ensure the vllm-tt-plugin package is installed and registers the TT platform.

    The TT platform lives in https://github.com/tenstorrent/vllm-tt-plugin, a
    standalone repository installed into the tt-metal venv by its own
    ``docs/install-vllm-tt.sh`` (which installs upstream vLLM first, then the
    plugin). Without it, vLLM starts with no ``tt`` platform registered.

    The probe runs unconditionally: it only needs the *installed* package, and
    there is no plugin source tree inside the vLLM checkout to key off. An
    earlier version gated this on ``$VLLM_DIR/plugins/vllm-tt-plugin`` existing
    -- the layout back when the plugin shipped inside the fork -- which meant
    validation silently no-opped for every standalone-plugin setup, i.e. in
    exactly the case it was meant to catch.

    Installing is deliberately not attempted here: the plugin's install script
    owns the vLLM version pin and its dependency overrides, so installing the
    plugin alone risks pairing it with an incompatible vLLM.
    """
    check_script = (
        "import vllm_tt_plugin; "
        "from importlib.metadata import entry_points; "
        "eps = {ep.name for ep in entry_points(group='vllm.platform_plugins')}; "
        "assert 'tt' in eps, "
        "f'tt platform plugin not registered in vllm.platform_plugins entry points, got: {eps}'"
    )
    return_code = run_command([str(venv_python), "-c", check_script], logger=logger)
    if return_code != 0:
        raise ValueError(
            "⛔ --local-server with inference engine vLLM requires the "
            "`vllm_tt_plugin` Python package (the TT platform plugin) "
            "to be installed in the tt-metal python environment and to register "
            "the `tt` entry under the `vllm.platform_plugins` entry-point group.\n"
            "Install it into the tt-metal python environment with:\n"
            "  git clone https://github.com/tenstorrent/vllm-tt-plugin.git\n"
            "  cd vllm-tt-plugin && source docs/install-vllm-tt.sh\n"
            "That script installs upstream vLLM (VLLM_TARGET_DEVICE=empty) plus "
            "the plugin; run it with the tt-metal venv activated. Verify with:\n"
            f"  {venv_python} -c 'import vllm_tt_plugin, ttnn; print(\"ok\")'\n"
            "See vllm-tt-metal/README.md for the full local installation steps."
        )
    logger.info(
        f"✅ validated vllm-tt-plugin install and `tt` platform_plugins entry "
        f"point registration with: {venv_python}"
    )


def validate_local_setup(model_spec, runtime_config, json_fpath):
    logger.info("Starting local setup validation")
    workflow_root_log_dir = get_default_workflow_root_log_dir()
    ensure_readwriteable_dir(workflow_root_log_dir)

    if (
        WorkflowType.from_string(runtime_config.workflow)
        in (WorkflowType.SERVER, WorkflowType.RELEASE)
    ) and (not runtime_config.skip_system_sw_validation):
        # check, and enforce if necessary, system software dependency versions
        venv_config = VENV_CONFIGS[WorkflowVenvType.SYSTEM_SOFTWARE_VALIDATION]
        venv_config.setup(model_spec=model_spec)

        # fmt: off
        cmd = [
            str(venv_config.venv_python),
            str(get_repo_root_path() / "workflows" / "run_system_software_validation.py"),
            "--runtime-model-spec-json", str(json_fpath),
        ]
        # fmt: on

        return_code = run_command(cmd, logger=logger)

        if return_code != 0:
            raise ValueError(
                "⛔ validating system software dependencies failed. See errors above for "
                "required version, and System Info section above for current system versions."
                "\nTo skip system software validation, use the flag: --skip-system-sw-validation"
            )
        logger.info("✅ validating system software dependencies completed")

    if (
        runtime_config.local_server
        and model_spec.inference_engine == InferenceEngine.VLLM.value
    ):
        _validate_local_vllm_installation(runtime_config)

    logger.info("✅ validating local setup completed")


def run_multihost_validation_subprocess(
    multihost_config, model_spec, json_fpath, dry_run=False
):
    """Run multihost validation via subprocess with dedicated venv.

    This aligns multihost validation with single-host validation pattern:
    - Uses SYSTEM_SOFTWARE_VALIDATION venv (with packaging library)
    - Runs run_multihost_validation.py as subprocess
    - Returns validated hosts list

    Args:
        multihost_config: MultiHostConfig object with hosts, paths, etc.
        model_spec: ModelSpec for system software version validation
        json_fpath: Path to runtime model spec JSON file
        dry_run: If True, skip directory existence and permission checks

    Returns:
        List of validated hostnames

    Raises:
        ValueError: If validation fails
    """
    venv_config = VENV_CONFIGS[WorkflowVenvType.SYSTEM_SOFTWARE_VALIDATION]
    venv_config.setup(model_spec=model_spec)

    cmd = [
        str(venv_config.venv_python),
        str(get_repo_root_path() / "workflows" / "run_multihost_validation.py"),
        "--hosts",
        ",".join(multihost_config.hosts),
        "--shared-storage-root",
        str(multihost_config.shared_storage_root),
        "--config-pkl-dir",
        str(multihost_config.config_pkl_dir),
        "--mpi-interface",
        multihost_config.mpi_interface,
        "--tt-smi-path",
        multihost_config.tt_smi_path,
    ]

    if json_fpath is not None:
        cmd.extend(["--runtime-model-spec-json", str(json_fpath)])

    if dry_run:
        cmd.append("--dry-run")

    return_code = run_command(cmd, logger=logger)

    if return_code != 0:
        raise ValueError(
            "⛔ Multi-host validation failed. See errors above.\n"
            "To skip system software validation, use the flag: --skip-system-sw-validation"
        )

    logger.info("✅ Multi-host validation completed")
    return multihost_config.hosts


def _try_fix_path_permissions_for_uid(path, uid, need_write=False):
    """Best-effort chmod to grant the target UID the required access bits.

    Determines which POSIX scope (owner/group/other) the UID falls into and
    adds read (+execute for directories, +write if need_write) bits for that
    scope.  Only succeeds when the current process has permission to chmod
    (i.e. current user owns the path or is root) -- no sudo required.

    Returns True if chmod was applied, False on failure.
    """
    path = Path(path)
    if not path.exists():
        return False

    st = path.stat()
    mode = st.st_mode
    gids = get_groups_for_uid(uid)

    if uid == st.st_uid:
        new_bits = stat.S_IRUSR
        if path.is_dir():
            new_bits |= stat.S_IXUSR
        if need_write:
            new_bits |= stat.S_IWUSR
    elif st.st_gid in gids:
        new_bits = stat.S_IRGRP
        if path.is_dir():
            new_bits |= stat.S_IXGRP
        if need_write:
            new_bits |= stat.S_IWGRP
    else:
        new_bits = stat.S_IROTH
        if path.is_dir():
            new_bits |= stat.S_IXOTH
        if need_write:
            new_bits |= stat.S_IWOTH

    target_mode = mode | new_bits
    if target_mode == mode:
        return False

    try:
        os.chmod(path, target_mode)
        logger.info(f"Fixed permissions on {path}: {oct(mode)} -> {oct(target_mode)}")
        return True
    except OSError as e:
        logger.debug(f"Cannot chmod {path}: {e}")
        return False


def validate_bind_mount_permissions(args):
    """Validate that --image-user UID can access bind-mounted host paths.

    Checks read permission for --host-hf-cache and --host-weights-dir (readonly mounts),
    and read+write permission for --host-volume (read-write mount).

    If a check fails, attempts to fix permissions via chmod (no sudo).
    Raises ValueError with actionable guidance if the fix is not possible.
    """
    uid = int(args.image_user)
    checks = []

    if args.host_volume:
        host_volume_path = Path(args.host_volume)
        if not host_volume_path.exists():
            logger.info(f"Creating host volume directory: {host_volume_path}")
            host_volume_path.mkdir(parents=True, exist_ok=True)
        checks.append(("--host-volume", args.host_volume, True))
    if args.host_hf_cache:
        checks.append(("--host-hf-cache", args.host_hf_cache, False))
    if getattr(args, "host_weights_dir", None):
        checks.append(("--host-weights-dir", args.host_weights_dir, False))

    for flag, host_path, need_write in checks:
        ok, reason = check_path_permissions_for_uid(
            host_path, uid, need_write=need_write
        )
        if not ok:
            _try_fix_path_permissions_for_uid(host_path, uid, need_write=need_write)
            ok, reason = check_path_permissions_for_uid(
                host_path, uid, need_write=need_write
            )
        if not ok:
            access_type = "read+write" if need_write else "read"
            raise ValueError(
                f"⛔ Bind mount permission check failed for {flag}={host_path}\n"
                f"  Container user (--image-user={uid}) needs {access_type} access.\n"
                f"  {reason}\n"
                f"  Fix: set --image-user to match the path owner UID, or adjust "
                f"permissions with chmod/chown on the host path."
            )
        logger.info(
            f"✅ Bind mount permission check passed for {flag}={host_path} "
            f"(uid={uid}, write={need_write})"
        )


def validate_local_server_paths(args):
    """Validate required host paths for --local-server execution."""
    if not args.local_server:
        return
    if not args.tt_metal_home:
        raise ValueError(
            "--local-server requires --tt-metal-home or TT_METAL_HOME to be set"
        )

    tt_metal_home = Path(args.tt_metal_home).expanduser().resolve()
    if not tt_metal_home.exists():
        raise ValueError(f"⛔ --tt-metal-home path does not exist: {tt_metal_home}")
    if not tt_metal_home.is_dir():
        raise ValueError(f"⛔ --tt-metal-home is not a directory: {tt_metal_home}")

    python_env_dir = _get_local_server_python_env_dir(args)
    venv_python = python_env_dir / "bin" / "python"
    build_lib_dir = tt_metal_home / "build" / "lib"
    entrypoint_path = (
        get_repo_root_path() / "vllm-tt-metal" / "src" / "run_vllm_api_server.py"
    )

    # No vLLM source dir is required: vLLM is an installed package in the tt-metal
    # venv, not a checkout. That it imports -- and that vllm-tt-plugin registers the
    # `tt` platform -- is checked by _validate_local_vllm_installation later.
    required_paths = [
        ("python venv interpreter", venv_python),
        ("tt-metal build/lib", build_lib_dir),
        ("local server entrypoint", entrypoint_path),
    ]
    for label, path in required_paths:
        if not path.exists():
            raise ValueError(f"⛔ Missing required {label}: {path}")

    if args.host_hf_cache:
        host_hf_cache = Path(args.host_hf_cache).expanduser().resolve()
        if not host_hf_cache.exists():
            raise ValueError(f"⛔ --host-hf-cache path does not exist: {host_hf_cache}")
        snapshot_dir = resolve_hf_snapshot_dir(
            args.runtime_model_spec["hf_weights_repo"], host_hf_cache
        )
        if snapshot_dir is None:
            raise ValueError(
                f"⛔ --host-hf-cache did not contain a cached snapshot for "
                f"{args.runtime_model_spec['hf_weights_repo']}: {host_hf_cache}"
            )

    if args.host_weights_dir:
        host_weights_dir = Path(args.host_weights_dir).expanduser().resolve()
        if not host_weights_dir.exists():
            raise ValueError(
                f"⛔ --host-weights-dir path does not exist: {host_weights_dir}"
            )


def validate_custom_weights(model_spec, runtime_config):
    """Fail fast on --custom-weights misconfiguration (source of bytes only).

    With --host-weights-dir the directory must exist and hold a recognizable
    weights layout. Without it the label must look like an HF repo id (org/name);
    Hub access is checked later during host setup.
    """
    custom_weights = getattr(runtime_config, "custom_weights", None)
    if not custom_weights:
        return

    host_weights_dir = getattr(runtime_config, "host_weights_dir", None)
    if host_weights_dir:
        # Local import avoids a circular import with setup_host.
        from workflows.setup_host import HostSetupManager

        weights_path = Path(host_weights_dir).expanduser().resolve()
        if not weights_path.exists():
            raise ValueError(
                f"⛔ --host-weights-dir path does not exist: {weights_path}"
            )
        manager = HostSetupManager(
            model_spec=model_spec,
            jwt_secret="",
            hf_token="",
            automatic=True,
            host_weights_dir=str(weights_path),
        )
        if not manager.check_model_weights_dir(weights_path):
            raise ValueError(
                f"⛔ --host-weights-dir={weights_path} does not contain a recognizable "
                "model weights layout (weights + tokenizer + params) for "
                f"--custom-weights '{custom_weights}'. Provide a directory with the "
                "model's safetensors/pth weights, tokenizer, and config files."
            )
        logger.info(
            f"✅ --custom-weights '{custom_weights}' will load local weights from "
            f"{weights_path}"
        )
    else:
        if "/" not in custom_weights:
            raise ValueError(
                f"⛔ --custom-weights='{custom_weights}' is not paired with "
                "--host-weights-dir, so it is treated as a HuggingFace repo id and "
                "must be of the form 'org/name'. Pass --host-weights-dir to load "
                "custom weights from local disk instead."
            )
        logger.info(
            f"✅ --custom-weights '{custom_weights}' will be downloaded from "
            f"HuggingFace as repo id '{model_spec.hf_weights_repo}'"
        )


def validate_setup(model_spec, runtime_config, json_fpath):
    """Top-level validation orchestrator called from run.py main().

    Runs all pre-flight validation checks in order:
    1. validate_runtime_args - CLI arg consistency and model/workflow support
    2. validate_custom_weights - --custom-weights source-of-bytes consistency
    3. validate_local_setup - system software dependencies
    4. validate_bind_mount_permissions - Docker bind mount UID access (docker-server only)
    """
    validate_runtime_args(model_spec, runtime_config)
    validate_custom_weights(model_spec, runtime_config)
    validate_local_setup(model_spec, runtime_config, json_fpath)
    if runtime_config.docker_server:
        validate_bind_mount_permissions(runtime_config)
    elif runtime_config.local_server:
        validate_local_server_paths(runtime_config)
