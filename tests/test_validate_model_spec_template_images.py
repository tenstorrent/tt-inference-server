from scripts.release.validate_model_spec_template_images import (
    format_missing_template_image_validation_error,
    validate_model_spec_template_images,
)
from workflows.model_spec import DeviceModelSpec, ImplSpec, ModelSpecTemplate
from workflows.workflow_types import DeviceTypes, InferenceEngine

TT_METAL_COMMIT = "a" * 40
VLLM_COMMIT = "1" * 7


def make_impl(name="demo-impl"):
    return ImplSpec(
        impl_id=name.replace("-", "_"),
        impl_name=name,
        repo_url="https://github.com/test/repo",
        code_path="models/test",
    )


def make_template(
    *,
    impl_name="demo-impl",
    weights=None,
    devices=None,
    docker_image="ghcr.io/tenstorrent/demo-release-image:tag",
    tt_metal_commit=TT_METAL_COMMIT,
    vllm_commit=VLLM_COMMIT,
):
    weights = weights or ["demo/model"]
    devices = devices or [DeviceTypes.N150]
    return ModelSpecTemplate(
        weights=weights,
        impl=make_impl(impl_name),
        tt_metal_commit=tt_metal_commit,
        vllm_commit=vllm_commit,
        inference_engine=InferenceEngine.VLLM.value,
        device_model_specs=[
            DeviceModelSpec(
                device=device,
                max_concurrency=16,
                max_context=4096,
                default_impl=True,
            )
            for device in devices
        ],
        docker_image=docker_image,
    )


def test_validate_model_spec_template_images_succeeds_when_all_images_exist():
    checked_images = []
    cache = {}
    template = make_template()

    def image_exists(image, cache=None):
        checked_images.append((image, cache))
        return True

    result = validate_model_spec_template_images(
        is_dev=False,
        check_image_exists=image_exists,
        templates=[template],
        cache=cache,
    )

    assert result.is_valid is True
    assert result.missing_images == ()
    assert checked_images == [("ghcr.io/tenstorrent/demo-release-image:tag", cache)]


def test_validate_model_spec_template_images_groups_missing_image_contexts():
    shared_image = "ghcr.io/tenstorrent/shared-release-image:tag"
    template_one = make_template(
        impl_name="demo-impl-a",
        weights=["demo/model-a"],
        docker_image=shared_image,
    )
    template_two = make_template(
        impl_name="demo-impl-b",
        weights=["demo/model-b"],
        docker_image=shared_image,
    )

    result = validate_model_spec_template_images(
        is_dev=False,
        check_image_exists=lambda image, cache=None: False,
        templates=[template_one, template_two],
    )

    assert result.is_valid is False
    assert len(result.missing_images) == 1
    assert result.missing_images[0].image == shared_image
    assert len(result.missing_images[0].references) == 2

    error_message = format_missing_template_image_validation_error(
        result,
        is_dev=False,
    )
    assert "scripts/release/README.md step 5B" in error_message
    assert "python3 scripts/build_docker_images.py --push --release" in error_message
    assert "weights=demo/model-a" in error_message
    assert "weights=demo/model-b" in error_message


def test_validate_model_spec_template_images_uses_dev_tags_in_dev_mode():
    checked_images = []
    template = make_template(
        docker_image="ghcr.io/tenstorrent/demo-release-image:tag",
    )

    def image_exists(image, cache=None):
        checked_images.append(image)
        return True

    result = validate_model_spec_template_images(
        is_dev=True,
        check_image_exists=image_exists,
        templates=[template],
    )

    assert result.is_valid is True
    assert checked_images == ["ghcr.io/tenstorrent/demo-dev-image:tag"]


def test_validate_model_spec_template_images_resolves_default_images_from_templates():
    checked_images = []
    template = make_template(
        docker_image=None,
        devices=[DeviceTypes.N150, DeviceTypes.DUAL_GALAXY],
    )

    def image_exists(image, cache=None):
        checked_images.append(image)
        return True

    result = validate_model_spec_template_images(
        is_dev=False,
        check_image_exists=image_exists,
        templates=[template],
    )

    assert result.is_valid is True
    assert len(checked_images) == 2
    assert any(
        image.startswith(
            "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:"
        )
        for image in checked_images
    )
    assert any(
        image.startswith(
            "ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-multihost-ubuntu-22.04-amd64:"
        )
        for image in checked_images
    )
