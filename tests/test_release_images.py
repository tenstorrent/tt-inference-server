import json
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import scripts.release.release_images as ri


def make_release_model_spec_export(model_specs):
    nested_specs = {}
    for spec in model_specs:
        nested_specs.setdefault(spec["hf_model_repo"], {})
        nested_specs[spec["hf_model_repo"]].setdefault(spec["device_type"], {})
        nested_specs[spec["hf_model_repo"]][spec["device_type"]].setdefault(
            spec["inference_engine"], {}
        )
        nested_specs[spec["hf_model_repo"]][spec["device_type"]][
            spec["inference_engine"]
        ][spec["impl_id"]] = spec
    return {
        "schema_version": "0.1.0",
        "release_version": "0.10.0",
        "model_specs": nested_specs,
    }


def make_exported_spec(model_id, hf_repo, device, docker_image):
    return {
        "model_id": model_id,
        "hf_model_repo": hf_repo,
        "device_type": device,
        "inference_engine": "vllm",
        "impl_id": "demo_impl",
        "docker_image": docker_image,
    }


def make_args(output_dir, **overrides):
    args = {
        "ci_artifacts_path": None,
        "models_ci_run_id": None,
        "out_root": None,
        "output_dir": str(output_dir),
        "release_model_spec_path": str(
            output_dir.parent.parent / "release_model_spec.json"
        ),
        "readme_path": "README.md",
        "validate_only": False,
        "no_build": False,
        "accept_images": True,
        "dry_run": False,
    }
    args.update(overrides)
    return Namespace(**args)


def test_build_release_scope_model_ids_expands_weights_and_devices():
    record = {
        "impl": "demo-impl",
        "impl_id": "demo_impl",
        "weights": ["demo/model-a", "demo/model-b"],
        "devices": ["N150", "N300"],
    }

    model_ids = ri._build_release_scope_model_ids([record])

    assert model_ids == {
        "id_demo_impl_model-a_n150",
        "id_demo_impl_model-a_n300",
        "id_demo_impl_model-b_n150",
        "id_demo_impl_model-b_n300",
    }


def test_run_from_args_validate_only_writes_summary_for_release_scoped_missing_images(
    tmp_path,
):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    output_dir.mkdir(parents=True)
    release_model_spec_path = tmp_path / "release_model_spec.json"
    target_image = "ghcr.io/tenstorrent/demo-release-image:tag"
    release_model_spec_path.write_text(
        json.dumps(
            make_release_model_spec_export(
                [
                    make_exported_spec(
                        "id_demo_impl_model-a_n150",
                        "demo/model-a",
                        "N150",
                        target_image,
                    )
                ]
            )
        )
    )
    (output_dir / "pre_release_models_diff.json").write_text(
        json.dumps(
            [
                {
                    "impl": "demo-impl",
                    "impl_id": "demo_impl",
                    "weights": ["demo/model-a"],
                    "devices": ["N150"],
                }
            ]
        )
    )
    args = make_args(
        output_dir,
        release_model_spec_path=str(release_model_spec_path),
        validate_only=True,
    )
    plan = SimpleNamespace(
        status=ri.IMAGE_STATUS_NEEDS_BUILD,
        target_image=target_image,
        ci_source_image=None,
    )

    with patch.object(ri, "check_docker_installed", return_value=True), patch.object(
        ri, "_load_optional_acceptance_warnings", return_value=[{"model_id": "demo"}]
    ), patch.object(
        ri, "_collect_generated_artifacts", return_value=["release_model_spec.json"]
    ), patch.object(ri, "_load_optional_ci_data", return_value={}), patch.object(
        ri, "merge_specs_with_ci_data", return_value={}
    ), patch.object(ri, "plan_image_action", return_value=plan), patch.object(
        ri, "write_output"
    ) as write_output_mock:
        assert ri.run_from_args(args) == 0

    write_output_mock.assert_called_once()
    assert write_output_mock.call_args.kwargs["images_to_build"] == [target_image]
    assert write_output_mock.call_args.kwargs["acceptance_warnings"] == [
        {"model_id": "demo"}
    ]


def test_run_from_args_fails_when_missing_image_is_outside_release_scope(tmp_path):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    output_dir.mkdir(parents=True)
    release_model_spec_path = tmp_path / "release_model_spec.json"
    release_model_spec_path.write_text(
        json.dumps(
            make_release_model_spec_export(
                [
                    make_exported_spec(
                        "id_demo_impl_model-a_n150",
                        "demo/model-a",
                        "N150",
                        "ghcr.io/tenstorrent/in-release:tag",
                    ),
                    make_exported_spec(
                        "id_demo_impl_model-b_n150",
                        "demo/model-b",
                        "N150",
                        "ghcr.io/tenstorrent/not-in-release:tag",
                    ),
                ]
            )
        )
    )
    (output_dir / "pre_release_models_diff.json").write_text(
        json.dumps(
            [
                {
                    "impl": "demo-impl",
                    "impl_id": "demo_impl",
                    "weights": ["demo/model-a"],
                    "devices": ["N150"],
                }
            ]
        )
    )
    args = make_args(output_dir, release_model_spec_path=str(release_model_spec_path))
    merged_spec = {
        "id_demo_impl_model-a_n150": SimpleNamespace(
            target_docker_image="ghcr.io/tenstorrent/in-release:tag"
        ),
        "id_demo_impl_model-b_n150": SimpleNamespace(
            target_docker_image="ghcr.io/tenstorrent/not-in-release:tag"
        ),
    }
    plans = [
        SimpleNamespace(
            status=ri.IMAGE_STATUS_NEEDS_BUILD,
            target_image="ghcr.io/tenstorrent/in-release:tag",
            ci_source_image=None,
        ),
        SimpleNamespace(
            status=ri.IMAGE_STATUS_NEEDS_BUILD,
            target_image="ghcr.io/tenstorrent/not-in-release:tag",
            ci_source_image=None,
        ),
    ]

    with patch.object(ri, "check_docker_installed", return_value=True), patch.object(
        ri, "_load_optional_acceptance_warnings", return_value=[]
    ), patch.object(ri, "_collect_generated_artifacts", return_value=[]), patch.object(
        ri, "_load_optional_ci_data", return_value={}
    ), patch.object(
        ri, "merge_specs_with_ci_data", return_value=merged_spec
    ), patch.object(ri, "plan_image_action", side_effect=plans), patch.object(
        ri, "write_output"
    ) as write_output_mock:
        assert ri.run_from_args(args) == 1

    write_output_mock.assert_not_called()


def test_run_from_args_promotes_ci_images_and_writes_summary(tmp_path):
    output_dir = tmp_path / "release_logs" / "v0.10.0"
    output_dir.mkdir(parents=True)
    release_model_spec_path = tmp_path / "release_model_spec.json"
    target_image = "ghcr.io/tenstorrent/demo-release-image:tag"
    source_image = "ghcr.io/tenstorrent/demo-ci-image:tag"
    release_model_spec_path.write_text(
        json.dumps(
            make_release_model_spec_export(
                [
                    make_exported_spec(
                        "id_demo_impl_model-a_n150",
                        "demo/model-a",
                        "N150",
                        target_image,
                    )
                ]
            )
        )
    )
    (output_dir / "pre_release_models_diff.json").write_text(
        json.dumps(
            [
                {
                    "impl": "demo-impl",
                    "impl_id": "demo_impl",
                    "weights": ["demo/model-a"],
                    "devices": ["N150"],
                }
            ]
        )
    )
    args = make_args(output_dir, release_model_spec_path=str(release_model_spec_path))
    merged_spec = {
        "id_demo_impl_model-a_n150": SimpleNamespace(target_docker_image=target_image)
    }
    plan = SimpleNamespace(
        status=ri.IMAGE_STATUS_COPY_FROM_CI,
        target_image=target_image,
        ci_source_image=source_image,
    )

    with patch.object(ri, "check_docker_installed", return_value=True), patch.object(
        ri, "check_crane_installed", return_value=True
    ), patch.object(
        ri, "_load_optional_acceptance_warnings", return_value=[]
    ), patch.object(
        ri, "_collect_generated_artifacts", return_value=["release_model_spec.json"]
    ), patch.object(
        ri, "_load_optional_ci_data", return_value={"id_demo_impl_model-a_n150": {}}
    ), patch.object(
        ri, "merge_specs_with_ci_data", return_value=merged_spec
    ), patch.object(ri, "plan_image_action", return_value=plan), patch.object(
        ri, "_prompt_to_continue"
    ), patch.object(
        ri, "copy_docker_image", return_value=True
    ) as copy_mock, patch.object(ri, "write_output") as write_output_mock:
        assert ri.run_from_args(args) == 0

    copy_mock.assert_called_once_with(source_image, target_image, dry_run=False)
    assert write_output_mock.call_args.kwargs["copied_images"] == {
        target_image: source_image
    }
