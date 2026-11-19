# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import fcntl
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import prebuilt_artifacts, tt_weights_download
from scripts.prebuilt_artifacts import BUILD_KEY_ENV_VARS, KERNEL_CACHE_DOCKERFILE, main
from workflows.kernel_cache import (
    KERNEL_CACHE_CONTAINER_DIR,
    KERNEL_CACHE_MANIFEST_CONTAINER_PATH,
)
from workflows.model_spec import MODEL_SPEC_CATALOG_FILES, load_templates_from_yaml
from workflows.utils import get_repo_root_path

MODEL_SPECS_DIR = get_repo_root_path() / "workflows" / "model_specs"
DOWNLOAD_SCRIPT = Path(tt_weights_download.__file__)
DOWNLOAD_SCRIPT_IN_IMAGE = "/home/container_app_user/tt_weights_download.py"
BASE_IMAGE = "registry.example/media-server:1.0-aaa-bbb-1"
KC_IMAGE = f"{BASE_IMAGE}-kc"
WEIGHTS_REPO = "example-org/model-tt-weights"


def write_files(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return root


def tree(root):
    files = (path for path in root.rglob("*") if path.is_file())
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in files}


def matches(path, patterns):
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns or ())


class FakeDocker:
    """Stands in for subprocess.run: answers image inspect and keeps the build context.

    The docker command named in `failing` exits 1.
    """

    def __init__(self):
        self.image_env, self.commands, self.build_context = [], [], None
        self.failing = None

    def __call__(self, command, capture_output=False, text=False, env=None):
        self.commands.append(command)
        if command[1] == self.failing:
            return SimpleNamespace(returncode=1, stdout="", stderr="daemon said no\n")
        stdout = "sha256:1111\n"
        if "{{json .Config.Env}}" in command:
            stdout = json.dumps(self.image_env)
        elif command[1] == "build":
            self.build_context = tree(Path(command[-1]))
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")


class FakeHub:
    """One in-memory Hugging Face repo behind the HfApi methods the CLI calls."""

    def __init__(self):
        self.private, self.tags, self.commits, self.sent = None, {}, [{}], []
        self.head, self.fetched = "0", []

    def _commit(self, files):
        self.commits.append(files)
        self.head = str(len(self.commits) - 1)
        return SimpleNamespace(oid=self.head)

    def files(self, revision=None):
        commit = self.tags.get(revision, revision)
        return self.commits[int(commit) if str(commit).isdigit() else -1]

    def create_repo(self, repo_id, private, exist_ok):
        self.private = private if self.private is None else self.private

    def model_info(self, repo_id, revision=None):
        commit = self.tags.get(revision, self.head)
        return SimpleNamespace(sha=commit, private=self.private)

    def list_repo_refs(self, repo_id):
        tags = [SimpleNamespace(name=n, target_commit=c) for n, c in self.tags.items()]
        return SimpleNamespace(tags=tags)

    def create_branch(self, repo_id, branch, exist_ok):
        assert exist_ok

    def create_tag(self, repo_id, tag, revision):
        self.tags[tag] = revision

    def file_exists(self, repo_id, filename, revision):
        return filename in self.files(revision)

    def delete_file(self, path_in_repo, repo_id, revision, commit_message):
        files = self.files().items()
        return self._commit(
            {path: data for path, data in files if path != path_in_repo}
        )

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, revision):
        return self._commit({**self.files(), path_in_repo: path_or_fileobj})

    def upload_folder(
        self,
        repo_id,
        folder_path,
        path_in_repo,
        revision,
        commit_message,
        allow_patterns,
        ignore_patterns=None,
        delete_patterns=None,
    ):
        added = {
            f"{path_in_repo}/{path}": data
            for path, data in tree(Path(folder_path)).items()
            if matches(path, allow_patterns) and not matches(path, ignore_patterns)
        }
        deleted = [f"{path_in_repo}/{pattern}" for pattern in delete_patterns or ()]
        files = self.files().items()
        kept = {path: data for path, data in files if not matches(path, deleted)}
        self.sent += added
        return self._commit({**kept, **added})

    def list_repo_tree(self, repo_id, recursive, revision):
        entries = [SimpleNamespace(path="tt_dit_cache")]
        for path, data in self.files(revision).items():
            large = path.endswith((".tensorbin", ".safetensors"))
            lfs = SimpleNamespace(sha256=hashlib.sha256(data).hexdigest())
            blob_id = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            entry = SimpleNamespace(path=path, size=len(data), blob_id=blob_id)
            entry.lfs = lfs if large else None
            entries.append(entry)
        return entries

    def hf_hub_download(self, repo_id, filename, revision, local_dir):
        content = self.files(revision)[filename]
        return str(write_files(Path(local_dir), {filename: content}) / filename)

    def snapshot_download(self, repo_id, revision, allow_patterns, local_dir):
        files = self.files(revision).items()
        wanted = {path: data for path, data in files if matches(path, allow_patterns)}
        self.fetched += wanted
        write_files(Path(local_dir), wanted)


IMAGE_FILES = {
    "tt-metal-cache4242/kernels/reader/111/brisc/brisc.elf": b"elf-1",
    "tt-metal-cache4242/kernels/writer/222/ncrisc/ncrisc.o": b"object",
    "tt-metal-cache/pch/00ff/umbrella.hpp.gch": b"pch",
}
NOT_IMAGE_FILES = {
    "tt-metal-cache4242/kernels/reader/111/brisc/brisc.elf.log": b"log",
    "tt-metal-cache4242/kernels/writer/222/ncrisc/ncrisc.9001_77.o": b"temp",
}
# Variables that the spec of a made-up model names as changing its kernels.
MODEL_VARIABLES = ("EXAMPLE_MODEL_TOPOLOGY", "EXAMPLE_MODEL_OUTPUT")
RECORDED_VARIABLES = (*BUILD_KEY_ENV_VARS, *MODEL_VARIABLES)
SET_VARIABLES = {
    "TT_METAL_INSPECTOR": "0",
    "TT_MM_THROTTLE_PERF": "5",
    "EXAMPLE_MODEL_TOPOLOGY": "linear",
    "EXAMPLE_MODEL_OUTPUT": "yuv420",
}
KERNEL_MANIFEST = {
    "schema_version": 1,
    "base_image": BASE_IMAGE,
    "kernel_cache_path": KERNEL_CACHE_CONTAINER_DIR,
    "build_keys": ["tt-metal-cache4242"],
    "kernel_hash_dirs": 2,
    "files": 3,
    "bytes": 14,
    "compiled_kernels_in_log": 2,
    "build_key_env": {**dict.fromkeys(RECORDED_VARIABLES), **SET_VARIABLES},
}


@pytest.fixture
def docker(monkeypatch):
    fake = FakeDocker()
    monkeypatch.setattr(prebuilt_artifacts.subprocess, "run", fake)
    return fake


@pytest.fixture
def kernel_cache(tmp_path):
    return write_files(tmp_path / "cache", {**IMAGE_FILES, **NOT_IMAGE_FILES})


def write_server_log(path, compiled):
    lines = [f"info | compiled k_{i}/{i}/ in 9 ms" for i in range(compiled)]
    path.write_text("\n".join(["info | opening mesh", *lines, "info | ready"]))
    return path


def manifest_argv(kernel_cache, tmp_path, compiled):
    """kernel-cache manifest arguments for a capture log with that many compile lines, and the output file."""
    out = tmp_path / "out" / "kernel_cache_manifest.json"
    argv = ["kernel-cache", "manifest", "--cache-dir", str(kernel_cache)]
    argv += ["--server-log", str(write_server_log(tmp_path / "server.log", compiled))]
    return argv + ["--base-image", BASE_IMAGE, "--out", str(out)], out


def test_kernel_cache_manifest_content_and_env_order(
    docker, kernel_cache, tmp_path, capsys
):
    docker.image_env = ["TT_MM_THROTTLE_PERF=4", "TT_METAL_INSPECTOR=1"]
    env = tmp_path / ".env"
    env.write_text(
        "HF_TOKEN=hf_secret\nTT_MM_THROTTLE_PERF=5\nEXAMPLE_MODEL_TOPOLOGY=ring"
    )
    env_vars = {"EXAMPLE_MODEL_TOPOLOGY": "linear", "EXAMPLE_MODEL_OUTPUT": "yuv420"}
    env_vars.update({"TT_METAL_INSPECTOR": 0, "TT_METAL_WATCHER": ""})
    metadata = {"prebuilt_artifacts": {"kernel_build_env": list(MODEL_VARIABLES)}}
    spec_json = tmp_path / "runtime_model_spec.json"
    spec_json.write_text(json.dumps({"env_vars": env_vars, "metadata": metadata}))
    argv, out = manifest_argv(kernel_cache, tmp_path, compiled=2)

    assert main(argv + ["--env-file", str(env), "--spec-json", str(spec_json)]) == 0

    manifest = json.loads(out.read_text())
    assert manifest.pop("created_utc").endswith("Z")
    assert manifest == KERNEL_MANIFEST
    assert list(manifest["build_key_env"]) == list(RECORDED_VARIABLES)
    assert "hf_secret" not in "".join(capsys.readouterr()) + out.read_text()


def test_kernel_cache_manifest_records_only_the_model_variables_the_spec_names(
    docker, kernel_cache, tmp_path, capsys
):
    env_vars = {"EXAMPLE_MODEL_TOPOLOGY": "linear", "TT_MM_THROTTLE_PERF": "5"}
    spec_json = tmp_path / "runtime_model_spec.json"
    spec_json.write_text(json.dumps({"env_vars": env_vars}))
    argv, out = manifest_argv(kernel_cache, tmp_path, compiled=2)

    assert main(argv + ["--spec-json", str(spec_json)]) == 0

    recorded = {**dict.fromkeys(BUILD_KEY_ENV_VARS), "TT_MM_THROTTLE_PERF": "5"}
    assert json.loads(out.read_text())["build_key_env"] == recorded
    note = "no metadata.prebuilt_artifacts.kernel_build_env in"
    assert note in capsys.readouterr().out


@pytest.mark.parametrize(
    "described, named",
    [
        ({"kernel_build_env": ["NOT A NAME"]}, "kernel_build_env"),
        ({"kernel_build_env": "EXAMPLE_MODEL_TOPOLOGY"}, "kernel_build_env"),
        ({"original_file": ["model_index.json"]}, "original_file"),
        ({"original_files": ["../model_index.json"]}, "original_files"),
        ({"original_shards": [{"index": "index.json"}]}, "original_shards"),
    ],
)
def test_kernel_cache_manifest_refuses_spec_metadata_it_cannot_read(
    docker, kernel_cache, tmp_path, capsys, described, named
):
    spec = {"env_vars": {}, "metadata": {"prebuilt_artifacts": described}}
    spec_json = tmp_path / "runtime_model_spec.json"
    spec_json.write_text(json.dumps(spec))
    argv, out = manifest_argv(kernel_cache, tmp_path, compiled=2)

    assert main(argv + ["--spec-json", str(spec_json)]) == 1

    error = capsys.readouterr().err
    assert error.endswith(f"this tool does not read as they are: {named}\n")
    assert not out.exists()


def test_kernel_cache_manifest_refuses_a_log_without_compile_lines(
    docker, kernel_cache, tmp_path, capsys
):
    argv, out = manifest_argv(kernel_cache, tmp_path, compiled=0)

    assert main(argv) == 1

    error = capsys.readouterr().err
    assert "has no compile line although the cache holds 2 kernel hash folders" in error
    assert "TT_METAL_LOG_KERNEL_COMPILE=0" in error
    assert not out.exists()


def build_image_argv(kernel_cache, tmp_path, manifest=KERNEL_MANIFEST):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest))
    argv = ["kernel-cache", "build-image", "--base-image", BASE_IMAGE]
    argv += ["--cache-dir", str(kernel_cache), "--manifest", str(manifest_file)]
    return argv + ["--tag", KC_IMAGE]


def test_build_image_stages_cache_files_and_bakes_set_variables(
    docker, kernel_cache, tmp_path, capsys
):
    assert main(build_image_argv(kernel_cache, tmp_path)) == 0

    assert "image-id: sha256:1111\n" in capsys.readouterr().out
    assert [command[1] for command in docker.commands] == ["build", "image"]
    staged = dict(docker.build_context)
    dockerfile = staged.pop("Dockerfile").decode()
    assert json.loads(staged.pop("kernel_cache_manifest.json")) == KERNEL_MANIFEST
    assert staged.pop("tt_weights_download.py") == DOWNLOAD_SCRIPT.read_bytes()
    assert staged == {
        f"kernel_cache/{path}": data for path, data in IMAGE_FILES.items()
    }
    assert dockerfile.startswith(KERNEL_CACHE_DOCKERFILE.read_text())
    owned = "--chown=container_app_user:container_app_user --chmod=755"
    manifest_in_image = KERNEL_CACHE_MANIFEST_CONTAINER_PATH
    copy_lines = [line for line in dockerfile.splitlines() if line.startswith("COPY ")]
    assert copy_lines == [
        f"COPY {owned} kernel_cache/ {KERNEL_CACHE_CONTAINER_DIR}/",
        f"COPY --chmod=644 kernel_cache_manifest.json {manifest_in_image}",
        f"COPY --chmod=755 tt_weights_download.py {DOWNLOAD_SCRIPT_IN_IMAGE}",
    ]
    baked = [f'ENV {name}="{value}"' for name, value in SET_VARIABLES.items()]
    env_lines = [line for line in dockerfile.splitlines() if line.startswith("ENV ")]
    assert env_lines == [f"ENV TT_METAL_CACHE={KERNEL_CACHE_CONTAINER_DIR}", *baked]


def test_build_image_bakes_the_variables_a_manifest_records_in_its_order(
    docker, kernel_cache, tmp_path
):
    """The manifest alone says which variables an image gets, whatever recorded them."""
    recorded = {
        "TT_METAL_INSPECTOR": "0",
        "EXAMPLE_MODEL_TOPOLOGY": "linear",
        "ANOTHER_MODEL_OPTION": 'a "quoted" $value',
        "TT_METAL_WATCHER": None,
    }
    manifest = {**KERNEL_MANIFEST, "build_key_env": recorded}

    assert main(build_image_argv(kernel_cache, tmp_path, manifest)) == 0

    dockerfile = docker.build_context["Dockerfile"].decode()
    env_lines = [line for line in dockerfile.splitlines() if line.startswith("ENV ")]
    assert env_lines[1:] == [
        'ENV TT_METAL_INSPECTOR="0"',
        'ENV EXAMPLE_MODEL_TOPOLOGY="linear"',
        'ENV ANOTHER_MODEL_OPTION="a \\"quoted\\" \\$value"',
    ]


@pytest.mark.parametrize(
    "failing, error",
    [
        ("build", "docker build failed"),
        ("image", "docker image inspect failed: daemon said no"),
    ],
)
def test_build_image_fails_when_a_docker_command_fails(
    docker, kernel_cache, tmp_path, capsys, failing, error
):
    docker.failing = failing

    assert main(build_image_argv(kernel_cache, tmp_path)) == 1

    output = capsys.readouterr()
    assert output.err == f"error: {error}\n"
    assert "image-id" not in output.out


def run_lone_script(folder, cache, original, python_options=()):
    """Run a copy of the download script with no repo around it and no token."""
    copy = shutil.copyfile(DOWNLOAD_SCRIPT, folder / DOWNLOAD_SCRIPT.name)
    command = [sys.executable, "-I", *python_options, str(copy)]
    command += ["--repo", WEIGHTS_REPO, "--revision", "main", "--variant", VARIANT]
    command += ["--dest-cache", str(cache), "--dest-original", str(original)]
    no_token = {name: value for name, value in os.environ.items() if name != "HF_TOKEN"}
    return subprocess.run(
        command, cwd=folder, env=no_token, capture_output=True, text=True
    )


def test_the_download_script_runs_without_the_repo(tmp_path):
    result = run_lone_script(tmp_path, "cache", "weights")

    assert (result.returncode, result.stderr) == (1, "error: HF_TOKEN is not set\n")
    assert os.listdir(tmp_path) == [DOWNLOAD_SCRIPT.name]


def test_build_image_refuses_a_tree_that_differs(
    docker, kernel_cache, tmp_path, capsys
):
    (kernel_cache / "tt-metal-cache4242/kernels/reader/111/added.o").write_bytes(b"x")

    assert main(build_image_argv(kernel_cache, tmp_path)) == 1

    error = capsys.readouterr().err
    assert "holds 4 files, 15 bytes; the manifest says 3 files, 14 bytes" in error
    assert docker.commands == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root can list any folder")
def test_build_image_refuses_a_cache_folder_it_cannot_list(
    docker, kernel_cache, tmp_path, capsys
):
    locked = kernel_cache / "tt-metal-cache4242/kernels/writer"
    locked.chmod(0)
    try:
        assert main(build_image_argv(kernel_cache, tmp_path)) == 1
    finally:
        locked.chmod(0o755)

    assert f"cannot read {locked}: Permission denied" in capsys.readouterr().err
    assert docker.commands == []


@pytest.mark.parametrize(
    "manifest, error",
    [
        ({}, "is not a manifest this command reads: no base_image"),
        ({**KERNEL_MANIFEST, "schema_version": 2}, "has schema_version 2"),
        ({**KERNEL_MANIFEST, "base_image": "other"}, "was made for base image other"),
        (
            {**KERNEL_MANIFEST, "build_key_env": {"A=B\nRUN x": "1"}},
            "does not map variable names to a value or null",
        ),
        (
            {**KERNEL_MANIFEST, "build_key_env": {"TT_METAL_WATCHER": 10}},
            "does not map variable names to a value or null",
        ),
    ],
)
def test_build_image_refuses_another_manifest(
    docker, kernel_cache, tmp_path, capsys, manifest, error
):
    assert main(build_image_argv(kernel_cache, tmp_path, manifest)) == 1

    assert error in capsys.readouterr().err
    assert docker.commands == []


@pytest.mark.parametrize("compiled, exit_code", [(0, 0), (25, 1)])
def test_check_log_fails_on_compiled_kernels(tmp_path, capsys, compiled, exit_code):
    argv = ["kernel-cache", "check-log", "--server-log"]
    argv.append(str(write_server_log(tmp_path / "server.log", compiled)))

    assert main(argv) == exit_code

    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith(f"compiled kernels: {compiled}")
    assert output[1:] == [f"k_{i}/{i}" for i in range(min(compiled, 20))]


# Two variants of a made-up model. They share the text encoder and differ in the
# transformer folder and in the adapter, a file from another repo.
VARIANT, SIBLING = "example-model-one", "example-model-two"
TRANSFORMER = {VARIANT: "transformer", SIBLING: "transformer_two"}
CONTAINER_CACHE = "/home/container_app_user/cache_root/tt_dit_cache"
SOURCE_COMMIT = "c0ffee12" + "0" * 32
SHARED = "model/text_encoder/mesh4x8_bf16"
OWN = {
    VARIANT: "model/transformer_lora-a/TP4_mesh4x8_bf16",
    SIBLING: "model/transformer_two_lora-b/TP4_mesh4x8_bf16",
}
NOT_LOADED = "model/vae_previous/mesh1x8_fp32"
EXTRA_FILE_OUTSIDE = {
    "repo": "example-org/adapters",
    "local_subdir": "..",
    "filename": "x",
}
CACHE_FILES = {
    f"{OWN[VARIANT]}/blocks.0.weight.tensorbin": b"abc",
    f"{OWN[VARIANT]}/blocks.0.bias.tensorbin": b"defgh",
    f"{OWN[SIBLING]}/blocks.0.weight.tensorbin": b"ref",
    f"{SHARED}/embed.weight.tensorbin": b"ijklmno",
    f"{NOT_LOADED}/conv.weight.tensorbin": b"stale",
}
SHARDS = {
    "model.language_model.layers.0.weight": "model-00001-of-00002.safetensors",
    "model.visual.blocks.0.weight": "model-00002-of-00002.safetensors",
}
INDEX = json.dumps({"weight_map": SHARDS}).encode()
LOG = f"../{VARIANT}/server.log"
OTHER_LAYOUT = {"manifests/old.json": b'{"variant": "old"}'}
NOT_READ_BY_A_CACHED_START = {
    "transformer/diffusion_pytorch_model.safetensors": b"big",
    "text_encoder/model-00001-of-00002.safetensors": b"big",
    "text_encoder/merges.txt": b"big",
    "adapters/another-adapter.safetensors": b"big",
}


def model_spec(variant, **metadata):
    """Runtime model spec JSON of a variant; metadata replaces keys of its metadata."""
    described = {
        "kernel_build_env": list(MODEL_VARIABLES),
        "original_files": [
            "model_index.json",
            f"{TRANSFORMER[variant]}/config.json",
            "vae/config.json",
            "vae/*",
            "audio_vae/*",
            "text_encoder/*.json",
            "tokenizer/**/*",
        ],
        "original_shards": [
            {
                "index": "text_encoder/model.safetensors.index.json",
                "tensor_prefix": "model.visual.",
            }
        ],
        "original_folders": [TRANSFORMER[variant], "text_encoder", "vae", "audio_vae"],
    }
    adapter = {"repo": "example-org/adapters", "local_subdir": "adapters"}
    adapter["filename"] = f"{variant}.safetensors"
    spec = {
        "hf_weights_repo": "example-org/original-model",
        "env_vars": {"TT_DIT_CACHE_DIR": CONTAINER_CACHE},
        "metadata": {"hf_extra_files": [adapter], "prebuilt_artifacts": described},
    }
    spec["metadata"].update(metadata)
    return json.dumps({"runtime_model_spec": spec}).encode()


def original_subset(variant):
    return {
        "model_index.json": b"{}",
        "tokenizer/tokenizer.json": b"{}",
        f"{TRANSFORMER[variant]}/config.json": b"{}",
        "text_encoder/config.json": b"{}",
        "text_encoder/preprocessor_config.json": b"{}",
        "text_encoder/model.safetensors.index.json": INDEX,
        "text_encoder/model-00002-of-00002.safetensors": b"vision",
        "vae/config.json": b"{}",
        "vae/diffusion_pytorch_model.safetensors": b"vae",
        "audio_vae/config.json": b"{}",
        "audio_vae/diffusion_pytorch_model.safetensors": b"audio",
        f"adapters/{variant}.safetensors": variant.encode(),
    }


def published(variant, original="original_subset"):
    """Repo paths of a variant: the cache directories its start used and its original subset."""
    subset = original_subset(variant).items()
    files = {f"{original}/{path}": data for path, data in subset}
    for path, data in CACHE_FILES.items():
        if path.startswith((OWN[variant], SHARED)):
            files[f"tt_dit_cache/{path}"] = data
    return files


@pytest.fixture
def cache_dir(tmp_path):
    markers = {f"{Path(path).parent}/cache_dict.json": b"" for path in CACHE_FILES}
    return write_files(tmp_path / "volume" / "tt_dit_cache", {**CACHE_FILES, **markers})


def upload_argv(cache_dir, variant=VARIANT, tt_metal_sha="a" * 40, server_sha="b" * 40):
    """Upload after a start of the variant: shared directory loaded, its own one written."""
    work = cache_dir.parent / variant
    files = {**original_subset(variant), **NOT_READ_BY_A_CACHED_START}
    for path in original_subset(variant):
        if not path.startswith("adapters/"):
            metadata = f".cache/huggingface/download/{path}.metadata"
            files[metadata] = f"{SOURCE_COMMIT}\netag\n0.0\n".encode()
    write_files(work / "weights", files)
    (work / "server.log").write_text(
        f"INFO | loading 1 cached weight tensors from '{CONTAINER_CACHE}/{SHARED}'...\n"
        "INFO | Cache does not exist. Loading PyTorch state dict.\n"
        f"INFO | Writing cache to '{CONTAINER_CACHE}/{OWN[variant]}'.\n"
    )
    (work / "spec.json").write_bytes(model_spec(variant))
    argv = ["tt-weights", "upload", "--cache-dir", str(cache_dir), "--variant", variant]
    argv += ["--original-dir", str(work / "weights"), "--repo", WEIGHTS_REPO]
    argv += ["--server-log", str(work / "server.log")]
    argv += ["--spec-json", str(work / "spec.json"), "--tt-metal-sha", tt_metal_sha]
    return argv + ["--server-sha", server_sha, "--manifest-out", str(work / "out.json")]


@pytest.fixture
def empty_hub(monkeypatch):
    hub = FakeHub()
    for script in (prebuilt_artifacts, tt_weights_download):
        monkeypatch.setattr(script, "hugging_face_api", lambda: hub)
    return hub


@pytest.fixture
def hub(empty_hub, cache_dir):
    assert main(upload_argv(cache_dir)) == 0
    return empty_hub


@pytest.mark.parametrize(
    "option, downloaded_with_hf, source",
    [([], True, SOURCE_COMMIT), (["--source-revision", "f" * 9], False, "f" * 9)],
)
def test_upload_publishes_what_the_log_names_and_tags_it_by_its_source(
    empty_hub, cache_dir, capsys, option, downloaded_with_hf, source
):
    hub = empty_hub
    argv = upload_argv(cache_dir) + option
    weights = cache_dir.parent / VARIANT / "weights"
    if not downloaded_with_hf:
        (weights / ".cache").rename(weights.parent / "moved")

    assert main(argv) == 0

    assert hub.private is True
    remote = dict(hub.files())
    manifest = json.loads(remote.pop(f"manifests/{VARIANT}.json"))
    assert remote == published(VARIANT)
    left_out = f"the log does not name: {OWN[SIBLING]}, {NOT_LOADED}\n"
    assert left_out in capsys.readouterr().out
    tag = f"{VARIANT}_src-{source[:8]}_mesh4x8_ttm-aaaaaaaa"
    assert hub.tags == {tag: hub.head}
    out = json.loads((weights.parent / "out.json").read_text())
    version = {"repo": WEIGHTS_REPO, "tag": tag, "commit": hub.head, "uploaded": True}
    assert out == {**manifest, **version}
    assert manifest.pop("created_utc").endswith("Z")
    sha256 = {path: hashlib.sha256(data).hexdigest() for path, data in remote.items()}
    assert manifest == {
        "schema_version": 1,
        "variant": VARIANT,
        "source": {"repo": "example-org/original-model", "revision": source},
        "mesh_shape": "4x8",
        "tt_metal_sha": "a" * 40,
        "server_sha": "b" * 40,
        "cache_directories": [
            {"path": SHARED, "files": 1, "bytes": 7},
            {"path": OWN[VARIANT], "files": 2, "bytes": 8},
        ],
        "original_files": [
            {"path": p, "bytes": len(d), "sha256": sha256[f"original_subset/{p}"]}
            for p, d in sorted(original_subset(VARIANT).items())
        ],
        "original_folders": ["transformer", "text_encoder", "vae", "audio_vae"],
    }


@pytest.mark.parametrize(
    "record, found",
    [
        (b"", f"1 of 11 files have no record, revisions found: {SOURCE_COMMIT}"),
        (b"another\n", f"revisions found: another, {SOURCE_COMMIT}"),
    ],
    ids=["file-without-a-record", "two-revisions"],
)
def test_upload_asks_for_the_source_revision_it_cannot_read(
    empty_hub, cache_dir, capsys, record, found
):
    argv = upload_argv(cache_dir)
    records = cache_dir.parent / VARIANT / "weights/.cache/huggingface/download"
    (records / "model_index.json.metadata").write_bytes(record)

    assert main(argv) == 1

    error = capsys.readouterr().err
    assert found in error and "with --source-revision" in error
    assert (empty_hub.private, empty_hub.head) == (None, "0")


def test_upload_reads_only_the_directories_the_log_names(empty_hub, cache_dir):
    tensor = cache_dir / SHARED / "embed.weight.tensorbin"
    (cache_dir / NOT_LOADED / "link").symlink_to(tensor)
    locked = write_files(cache_dir / "another_cache", {"file": b""})
    locked.chmod(0)
    try:
        assert main(upload_argv(cache_dir)) == 0
    finally:
        locked.chmod(0o755)

    remote = set(empty_hub.files()) - {f"manifests/{VARIANT}.json"}
    assert remote == set(published(VARIANT))


def test_upload_refuses_a_cache_dir_that_is_not_a_directory(
    empty_hub, cache_dir, capsys
):
    argv = upload_argv(cache_dir)
    argv[argv.index("--cache-dir") + 1] += "-mistyped"

    assert main(argv) == 1

    assert "tt_dit_cache-mistyped is not a directory\n" in capsys.readouterr().err
    assert empty_hub.private is None


def test_upload_stores_shared_paths_once_and_replaces_only_its_own(hub, cache_dir):
    hub.sent.clear()
    assert main(upload_argv(cache_dir, SIBLING)) == 0
    assert set(hub.sent) == set(published(SIBLING)) - set(published(VARIANT))
    own = cache_dir / OWN[SIBLING]
    (own / "blocks.0.weight.tensorbin").rename(own / "blocks.0.scale.tensorbin")
    nested = f"tt_dit_cache/{OWN[SIBLING]}/host_coords_r0-3_c0-7/a.tensorbin"
    hub.files()[nested] = b"a multi-host capture of the same component"
    assert main(upload_argv(cache_dir, SIBLING, tt_metal_sha="c" * 40)) == 0
    remote = {path for path in hub.files() if not path.startswith("manifests/")}
    renamed = {path.replace("0.weight", "0.scale") for path in published(SIBLING)}
    assert remote == renamed | set(published(VARIANT)) | {nested}


def test_upload_does_not_tag_a_commit_that_holds_other_content(
    empty_hub, cache_dir, capsys
):
    hub = empty_hub
    upload_file = hub.upload_file

    def another_upload_writes(**arguments):
        hub.files()[f"tt_dit_cache/{SHARED}/embed.weight.tensorbin"] = b"7 bytes"
        return upload_file(**arguments)

    hub.upload_file = another_upload_writes

    assert main(upload_argv(cache_dir)) == 1

    assert "something else changed main" in capsys.readouterr().err
    assert hub.tags == {}


def test_replace_shared_removes_the_manifests_that_list_the_path(hub, cache_dir):
    assert main(upload_argv(cache_dir, SIBLING)) == 0
    tagged = {tag: dict(hub.files(tag)) for tag in hub.tags}
    (cache_dir / SHARED / "embed.weight.tensorbin").write_bytes(b"converted anew")
    argv = upload_argv(cache_dir, tt_metal_sha="c" * 40) + ["--replace-shared"]

    assert main(argv) == 0

    head = hub.files()
    assert head[f"tt_dit_cache/{SHARED}/embed.weight.tensorbin"] == b"converted anew"
    manifests = {path for path in head if path.startswith("manifests/")}
    assert manifests == {f"manifests/{VARIANT}.json"}
    assert tagged.items() <= {tag: hub.files(tag) for tag in hub.tags}.items()


def write(path, content):
    return lambda cache_dir, hub: (cache_dir / path).write_bytes(content)


def remove(path):
    return lambda cache_dir, hub: (cache_dir / path).unlink()


def upload_again(hub, cache_dir, capsys, tt_metal, change, options=()):
    """Exit code and output of a second VARIANT upload, made after a change and next to SIBLING.

    The Hub must be left as it was.
    """
    assert main(upload_argv(cache_dir, SIBLING)) == 0
    argv = upload_argv(cache_dir, tt_metal_sha=tt_metal * 40, server_sha="d" * 40)
    change(cache_dir, hub)
    before = (hub.head, dict(hub.tags))
    exit_code = main(argv + list(options))
    assert (hub.head, hub.tags) == before
    return exit_code, "".join(capsys.readouterr())


MANIFEST = f"manifests/{VARIANT}.json"
FIRST_TAG = f"{VARIANT}_src-c0ffee12_mesh4x8_ttm-aaaaaaaa"
OWN_TENSOR = f"{OWN[VARIANT]}/blocks.0.weight.tensorbin"
OWN_ORIGINAL = f"{TRANSFORMER[VARIANT]}/config.json"


def upload_out(cache_dir, variant=VARIANT):
    return json.loads((cache_dir.parent / variant / "out.json").read_text())


def keep_the_fields_a_download_reads(cache_dir, hub):
    """Leaves every published manifest of VARIANT without its other fields."""
    fields = ("schema_version", *tt_weights_download.WEIGHTS_MANIFEST_FIELDS)
    for files in hub.commits:
        if MANIFEST in files:
            manifest = json.loads(files[MANIFEST])
            files[MANIFEST] = json.dumps({f: manifest[f] for f in fields}).encode()


def tag_the_version_twice(cache_dir, hub):
    hub.tags[f"{VARIANT}_a"] = hub.tags[FIRST_TAG]


UPLOADS_OF_PUBLISHED_CONTENT = {
    "another-tt-metal-commit": ("c", [], lambda cache_dir, hub: None),
    "same-tag-name": ("a", [], tag_the_version_twice),
    "another-source-revision": (
        "a",
        ["--source-revision", "f" * 9],
        lambda cache_dir, hub: None,
    ),
    "manifest-with-the-fields-a-download-reads": (
        "c",
        [],
        keep_the_fields_a_download_reads,
    ),
    "directory-the-log-does-not-name": (
        "c",
        [],
        write(f"{NOT_LOADED}/x.tensorbin", b""),
    ),
}


@pytest.mark.parametrize(
    "tt_metal, options, change",
    UPLOADS_OF_PUBLISHED_CONTENT.values(),
    ids=UPLOADS_OF_PUBLISHED_CONTENT.keys(),
)
def test_upload_of_published_content_sends_nothing_and_names_that_version(
    hub, cache_dir, capsys, tt_metal, options, change
):
    commit = hub.tags[FIRST_TAG]

    exit_code, output = upload_again(hub, cache_dir, capsys, tt_metal, change, options)

    assert exit_code == 0
    assert [line for line in output.splitlines() if "unchanged" in line] == [
        f"unchanged: {WEIGHTS_REPO} holds this content as tag {FIRST_TAG} at "
        f"commit {commit}; nothing was uploaded"
    ]
    version = {"repo": WEIGHTS_REPO, "tag": FIRST_TAG, "commit": commit}
    published_manifest = json.loads(hub.files(FIRST_TAG)[MANIFEST])
    assert upload_out(cache_dir) == {**published_manifest, **version, "uploaded": False}


def content_hash(files):
    """sha256 over the sha256 and repo path of each file, in path order."""
    lines = [
        f"{hashlib.sha256(data).hexdigest()}  {path}\n"
        for path, data in sorted(files.items())
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


NEW_CONTENT = {
    "converted-file-of-the-same-size": (
        f"tt_dit_cache/{OWN_TENSOR}",
        b"XYZ",
        {
            f"tt_dit_cache/{path}"
            for path in CACHE_FILES
            if path.startswith(OWN[VARIANT])
        },
    ),
    "original-file-of-the-same-size": (
        f"original_subset/{OWN_ORIGINAL}",
        b"[]",
        {f"original_subset/{OWN_ORIGINAL}"},
    ),
}


@pytest.mark.parametrize("tt_metal", ["c", "a"], ids=["new-tag-name", "tag-name-taken"])
@pytest.mark.parametrize(
    "path, data, sent", NEW_CONTENT.values(), ids=NEW_CONTENT.keys()
)
def test_upload_of_new_content_makes_a_version_and_moves_no_tag(
    hub, cache_dir, capsys, path, data, sent, tt_metal
):
    first = (hub.tags[FIRST_TAG], dict(hub.files(FIRST_TAG)))
    argv = upload_argv(cache_dir, tt_metal_sha=tt_metal * 40)
    roots = {"tt_dit_cache": ".", "original_subset": f"../{VARIANT}/weights"}
    folder, _, relative = path.partition("/")
    (cache_dir / roots[folder] / relative).write_bytes(data)
    hub.sent.clear()

    assert main(argv) == 0

    output = capsys.readouterr().out
    tag = f"{VARIANT}_src-c0ffee12_mesh4x8_ttm-{tt_metal * 8}"
    if tag == FIRST_TAG:
        tag += "_" + content_hash({**published(VARIANT), path: data})[:8]
        taken = (
            f"tag: {FIRST_TAG} holds other content: the same inputs gave other "
            f"bytes; this version is tagged {tag}\n"
        )
        assert taken in output
    assert "unchanged" not in output and f"tag: {tag}\n" in output
    assert set(hub.sent) == sent
    assert hub.tags == {FIRST_TAG: first[0], tag: hub.head}
    assert hub.files(FIRST_TAG) == first[1]
    assert hub.files(tag)[path] == data
    out = upload_out(cache_dir)
    assert (out["tag"], out["commit"], out["uploaded"]) == (tag, hub.head, True)


def another_original_folder(cache_dir, hub):
    path = cache_dir.parent / VARIANT / "spec.json"
    spec = json.loads(path.read_text())
    described = spec["runtime_model_spec"]["metadata"]["prebuilt_artifacts"]
    described["original_folders"].append("tokenizer")
    path.write_text(json.dumps(spec))


PUBLISHED_FILES_LISTED_OTHERWISE = {
    "fewer-directories": (
        write(LOG, f"'{CONTAINER_CACHE}/{OWN[VARIANT]}'".encode()),
        "cache_directories",
        [{"path": OWN[VARIANT], "files": 2, "bytes": 8}],
    ),
    "another-folder": (
        another_original_folder,
        "original_folders",
        ["transformer", "text_encoder", "vae", "audio_vae", "tokenizer"],
    ),
}


@pytest.mark.parametrize(
    "change, field, listed",
    PUBLISHED_FILES_LISTED_OTHERWISE.values(),
    ids=PUBLISHED_FILES_LISTED_OTHERWISE.keys(),
)
def test_upload_of_published_files_under_another_list_makes_a_version(
    hub, cache_dir, change, field, listed
):
    argv = upload_argv(cache_dir, tt_metal_sha="c" * 40)
    change(cache_dir, hub)
    hub.sent.clear()

    assert main(argv) == 0

    out = upload_out(cache_dir)
    assert (out["uploaded"], out["commit"], hub.sent) == (True, hub.head, [])
    assert out[field] == listed and len(hub.tags) == 2


def test_upload_finds_the_version_that_holds_the_content_among_several(hub, cache_dir):
    tensor = cache_dir / OWN_TENSOR
    tensor.write_bytes(b"XYZ")
    assert main(upload_argv(cache_dir, tt_metal_sha="c" * 40)) == 0
    head, versions = hub.head, dict(hub.tags)
    assert len(versions) == 2

    for data, tt_metal in ((b"abc", "a"), (b"XYZ", "c")):
        tensor.write_bytes(data)
        assert main(upload_argv(cache_dir, tt_metal_sha="e" * 40)) == 0
        tag = f"{VARIANT}_src-c0ffee12_mesh4x8_ttm-{tt_metal * 8}"
        out = upload_out(cache_dir)
        assert (out["tag"], out["commit"]) == (tag, versions[tag])
        assert (out["uploaded"], out["tt_metal_sha"]) == (False, tt_metal * 40)

    assert (hub.head, hub.tags) == (head, versions)


def test_upload_reads_each_file_once_for_its_sha256(hub, cache_dir, monkeypatch):
    read, file_hash = [], tt_weights_download.file_hash

    def counted(path, digest=None):
        read.append((path, digest is None))
        return file_hash(path, digest)

    for script in (prebuilt_artifacts, tt_weights_download):
        monkeypatch.setattr(script, "file_hash", counted)
    (cache_dir / OWN_TENSOR).write_bytes(b"XYZ")

    assert main(upload_argv(cache_dir, tt_metal_sha="c" * 40)) == 0

    whole = [path for path, sha256 in read if sha256]
    assert len(whole) == len(set(whole)) == len(published(VARIANT))
    assert {path.suffix for path, sha256 in read if not sha256} == {".json"}


def test_upload_prefers_a_tag_of_the_variant_to_a_tag_of_another(hub, cache_dir):
    assert main(upload_argv(cache_dir, SIBLING)) == 0
    own = dict(hub.tags).popitem()
    (cache_dir / OWN_TENSOR).write_bytes(b"XYZ")
    assert main(upload_argv(cache_dir, tt_metal_sha="c" * 40)) == 0
    later = f"{VARIANT}_src-c0ffee12_mesh4x8_ttm-cccccccc"
    assert f"manifests/{SIBLING}.json" in hub.files(later) and later < own[0]

    assert main(upload_argv(cache_dir, SIBLING, tt_metal_sha="e" * 40)) == 0

    out = upload_out(cache_dir, SIBLING)
    assert (out["tag"], out["commit"], out["uploaded"]) == (*own, False)


def another_layout_on_the_branch(cache_dir, hub):
    hub.files().update(OTHER_LAYOUT)
    write(f"{OWN[VARIANT]}/x.tensorbin", b"")(cache_dir, hub)


def both_tag_names_taken(cache_dir, hub):
    write(OWN_TENSOR, b"XYZ")(cache_dir, hub)
    tag = f"{VARIANT}_src-c0ffee12_mesh4x8_ttm-cccccccc"
    content = {**published(VARIANT), f"tt_dit_cache/{OWN_TENSOR}": b"XYZ"}
    hub.tags.update({tag: "0", f"{tag}_{content_hash(content)[:8]}": "0"})


UPLOADS_REFUSED = {
    "shared-directory-changed": (
        write(f"{SHARED}/x.tensorbin", b""),
        f"the manifest of {SIBLING} lists",
    ),
    "temp-file": (write(f"{SHARED}/x.tmp.42.0", b""), "1 temp file(s)"),
    "no-marker": (remove(f"{SHARED}/cache_dict.json"), "no cache_dict.json"),
    "symbolic-link": (
        lambda cache_dir, hub: (cache_dir / SHARED / "link").symlink_to("/"),
        "symbolic links",
    ),
    "spec-without-the-metadata": (
        write(f"../{VARIANT}/spec.json", model_spec(VARIANT, prebuilt_artifacts={})),
        "no metadata.prebuilt_artifacts.original_files in the model spec",
    ),
    "spec-with-an-extra-file-outside": (
        write(
            f"../{VARIANT}/spec.json",
            model_spec(VARIANT, hf_extra_files=[EXTRA_FILE_OUTSIDE]),
        ),
        "names files outside the checkpoint folder: ../x",
    ),
    "index-without-the-prefix": (
        write(f"../{VARIANT}/weights/text_encoder/model.safetensors.index.json", b"{}"),
        "no model.visual.* weights in",
    ),
    "log-names-no-directory": (write(LOG, b"ready"), "names no directory under"),
    "log-names-a-missing-directory": (
        write(LOG, f"'{CONTAINER_CACHE}/gone'".encode()),
        "gone: missing or empty",
    ),
    "log-names-a-path-outside": (
        write(LOG, f"'{CONTAINER_CACHE}/../{VARIANT}/weights/vae'".encode()),
        "weights/vae: outside",
    ),
    "original-file-missing": (
        remove(f"../{VARIANT}/weights/vae/config.json"),
        "reads: vae/config.json",
    ),
    "repo-not-private": (
        lambda cache_dir, hub: setattr(hub, "private", False),
        "is not private",
    ),
    "manifest-of-another-layout": (another_layout_on_the_branch, "old.json at main"),
    "both-tag-names-taken": (
        both_tag_names_taken,
        "with other content; tags are not moved",
    ),
}


@pytest.mark.parametrize(
    "change, message", UPLOADS_REFUSED.values(), ids=UPLOADS_REFUSED.keys()
)
def test_upload_that_is_refused_does_not_change_the_hub(
    hub, cache_dir, capsys, change, message
):
    exit_code, output = upload_again(hub, cache_dir, capsys, "c", change)

    assert message in output
    assert exit_code == 1


@pytest.mark.parametrize("catalog", ["dev", "prod"])
def test_the_tool_reads_the_metadata_of_every_catalog_spec_that_has_it(
    tmp_path, catalog
):
    spec_json = tmp_path / "runtime_model_spec.json"
    for name in MODEL_SPEC_CATALOG_FILES:
        path = MODEL_SPECS_DIR / catalog / name
        for template in load_templates_from_yaml(path, env=catalog):
            for spec in template.expand_to_specs():
                described = spec.metadata.get("prebuilt_artifacts")
                if described is None:
                    continue
                serialized = {"runtime_model_spec": spec.get_serialized_dict()}
                spec_json.write_text(json.dumps(serialized))
                read = prebuilt_artifacts.spec_prebuilt_artifacts(spec_json)
                assert read == described and described["original_files"]
                extra_files = spec.metadata.get("hf_extra_files", [])
                read = prebuilt_artifacts.spec_extra_files(spec_json)
                assert len(read) == len(extra_files)


RECORDS = "tt_dit_cache/.tt_weights_complete"


class NoHub:
    """Hub client of a run that must not reach the Hub: any call on it fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"the Hub was asked for {name}")


def download(dest, variant=VARIANT, revision="main", repo=WEIGHTS_REPO):
    argv = ["--repo", repo, "--revision", revision, "--variant", variant]
    argv += ["--dest-cache", str(dest / "tt_dit_cache")]
    return tt_weights_download.main(argv + ["--dest-original", str(dest / "weights")])


def hub_reasons(capsys):
    """Why each run since the last read of the output asked the Hub."""
    lines = capsys.readouterr().out.splitlines()
    return [line[13:] for line in lines if line.startswith("hub: needed, ")]


def test_download_marks_one_variant_and_keeps_it_when_another_fails(
    hub, cache_dir, tmp_path
):
    assert main(upload_argv(cache_dir, SIBLING)) == 0
    nested = f"tt_dit_cache/{OWN[VARIANT]}/host_coords_r0-3_c0-7/a.tensorbin"
    hub.files()[nested] = b"a multi-host capture of the same component"
    dest = tmp_path / "deployment"

    assert download(dest) == 0

    assert sorted(hub.fetched) == sorted(published(VARIANT))
    expected = published(VARIANT, original="weights")
    for directory in (OWN[VARIANT], SHARED):
        expected[f"tt_dit_cache/{directory}/cache_dict.json"] = b""
    files = tree(dest)
    record = json.loads(files.pop(f"{RECORDS}/{VARIANT}.json"))
    assert files == expected
    named = [record[name] for name in ("repo", "variant", "revision", "commit")]
    assert named == [WEIGHTS_REPO, VARIANT, "main", hub.head]
    assert record["manifest"] == json.loads(hub.files()[f"manifests/{VARIANT}.json"])

    hub.fetched.clear()
    del hub.files()[f"tt_dit_cache/{OWN[SIBLING]}/blocks.0.weight.tensorbin"]
    shutil.rmtree(dest / RECORDS)
    assert download(dest) == 0
    assert download(dest, SIBLING) == 1

    assert not set(hub.fetched) & set(published(VARIANT))
    applied = {p: d for p, d in tree(dest).items() if ".tt_weights_" not in p}
    assert applied == expected
    assert json.loads((dest / RECORDS / f"{VARIANT}.json").read_text()) == record
    assert not (dest / RECORDS / f"{SIBLING}.json").exists()


def test_download_refuses_a_destination_another_download_holds(hub, tmp_path, capsys):
    dest = tmp_path / "deployment"
    (dest / "tt_dit_cache").mkdir(parents=True)
    lock = os.open(dest / "tt_dit_cache", os.O_RDONLY)
    fcntl.flock(lock, fcntl.LOCK_EX)

    assert download(dest) == 1

    os.close(lock)
    assert "another download is writing" in capsys.readouterr().err
    assert tree(dest) == {}


def test_download_refuses_a_directory_with_other_files_and_changes_nothing(
    hub, tmp_path, capsys
):
    dest = tmp_path / "deployment"
    assert download(dest) == 0
    hub.fetched.clear()
    other = {"renamed.tensorbin": b"converted by another tt-metal commit"}
    write_files(dest / "tt_dit_cache" / OWN[VARIANT], other)
    before = tree(dest)

    assert download(dest) == 1

    error = capsys.readouterr().err
    assert f"{OWN[VARIANT]}: 1 file(s), for example renamed.tensorbin" in error
    assert (tree(dest), hub.fetched) == (before, [])


@pytest.mark.parametrize("unsafe", ["../outside", "/outside", ""])
def test_download_refuses_a_manifest_with_an_unsafe_path(hub, tmp_path, capsys, unsafe):
    manifest = json.loads(hub.files()[f"manifests/{VARIANT}.json"])
    manifest["cache_directories"][0]["path"] = unsafe
    hub.files()[f"manifests/{VARIANT}.json"] = json.dumps(manifest).encode()
    dest = tmp_path / "deployment"

    assert download(dest) == 1

    assert f"unsafe path in the manifest: {unsafe}\n" in capsys.readouterr().err
    assert (tree(dest), hub.fetched) == ({}, [])


@pytest.mark.parametrize(
    "path, content",
    [
        (f"tt_dit_cache/{OWN[VARIANT]}/blocks.0.bias.tensorbin", None),
        ("original_subset/vae/config.json", b"[]"),
    ],
)
def test_download_failure_leaves_no_marker(hub, tmp_path, capsys, path, content):
    del hub.files()[path]
    if content:
        hub.files()[path] = content
    dest = tmp_path / "deployment"
    previous = {"embed.weight.tensorbin": b"old", "cache_dict.json": b""}
    write_files(dest / "tt_dit_cache" / SHARED, previous)

    assert download(dest) == 1

    assert "the download was not applied" in capsys.readouterr().err
    assert not list(dest.rglob("cache_dict.json"))
    assert tree(dest / "tt_dit_cache" / SHARED) == {"embed.weight.tensorbin": b"old"}
    assert not (dest / "weights").exists()
    assert not (dest / RECORDS).exists()


def test_download_into_a_complete_destination_needs_no_hub_and_no_token(
    hub, tmp_path, monkeypatch, capsys
):
    dest = tmp_path / "deployment"
    assert download(dest) == 0
    before = tree(dest)
    capsys.readouterr()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(tt_weights_download, "hugging_face_api", NoHub)

    assert download(dest) == 0
    assert download(dest, revision=hub.head) == 0

    complete = (
        f"complete: {dest / 'tt_dit_cache'} holds {VARIANT} of {WEIGHTS_REPO} at "
        f"commit {hub.head}; 2 cache directories and 12 original files re-checked "
        "against the saved manifest; the Hub was not contacted"
    )
    assert capsys.readouterr().out.splitlines() == [complete, complete]
    assert tree(dest) == before


def test_the_lone_download_script_checks_a_complete_destination(hub, tmp_path):
    dest = tmp_path / "deployment"
    assert download(dest) == 0
    lone = tmp_path / "lone"
    lone.mkdir()

    result = run_lone_script(
        lone, dest / "tt_dit_cache", dest / "weights", ["-X", "importtime"]
    )

    assert result.returncode == 0
    assert result.stdout.startswith("complete: ") and result.stdout.count("\n") == 1
    assert "import time:" in result.stderr and "huggingface_hub" not in result.stderr


def test_download_of_another_revision_asks_the_hub(hub, tmp_path, capsys):
    dest = tmp_path / "deployment"
    assert download(dest) == 0
    record = dest / RECORDS / f"{VARIANT}.json"
    (tag,) = hub.tags
    capsys.readouterr()

    assert download(dest, revision=tag) == 0

    output = capsys.readouterr().out
    reason = f"{record} records revision main (commit {hub.head}), not {tag}"
    assert output.startswith(f"hub: needed, {reason}\nrevision: {tag} is commit ")
    assert "keep: 2 cache directories and 12 original files" in output
    assert json.loads(record.read_text())["revision"] == tag


def test_download_leaves_the_markers_it_keeps_alone(hub, tmp_path, monkeypatch, capsys):
    """The kept markers are read-only files whose times this user cannot set."""
    dest = tmp_path / "deployment"
    assert download(dest) == 0
    before = tree(dest)
    shutil.rmtree(dest / RECORDS)
    markers = sorted(dest.rglob("cache_dict.json"))
    for marker in markers:
        marker.chmod(0o444)
    stats = [(marker.stat().st_ino, marker.stat().st_mtime_ns) for marker in markers]
    set_times = os.utime

    def utime(path, *args, **kwargs):
        if Path(path) in markers:
            raise PermissionError(f"{path} belongs to another user")
        return set_times(path, *args, **kwargs)

    monkeypatch.setattr(os, "utime", utime)
    hub.fetched.clear()
    capsys.readouterr()

    assert download(dest) == 0

    assert "keep: 2 cache directories and 12 original files" in capsys.readouterr().out
    assert [(m.stat().st_ino, m.stat().st_mtime_ns) for m in markers] == stats
    assert (tree(dest), hub.fetched) == (before, [])


CHANGES_AFTER_A_DOWNLOAD = {
    "tensor-removed": (
        f"tt_dit_cache/{OWN[VARIANT]}/blocks.0.bias.tensorbin",
        None,
        "does not hold the recorded file names and sizes",
    ),
    "tensor-of-another-size": (
        f"tt_dit_cache/{SHARED}/embed.weight.tensorbin",
        b"cut",
        "does not hold the recorded file names and sizes",
    ),
    "marker-removed": (
        f"tt_dit_cache/{SHARED}/cache_dict.json",
        None,
        "has no cache_dict.json",
    ),
    "original-file-of-the-same-size": (
        "weights/vae/config.json",
        b"[]",
        "is missing or not the recorded file",
    ),
    "original-file-removed": (
        f"weights/adapters/{VARIANT}.safetensors",
        None,
        "is missing or not the recorded file",
    ),
    "record-cut-short": (
        f"{RECORDS}/{VARIANT}.json",
        b"{",
        "is not a record this script reads",
    ),
}


@pytest.mark.parametrize(
    "path, content, reason",
    CHANGES_AFTER_A_DOWNLOAD.values(),
    ids=CHANGES_AFTER_A_DOWNLOAD.keys(),
)
def test_download_repairs_a_recorded_destination_that_changed(
    hub, tmp_path, capsys, path, content, reason
):
    dest = tmp_path / "deployment"
    assert download(dest) == 0
    before = tree(dest)
    (dest / path).unlink()
    if content:
        (dest / path).write_bytes(content)
    capsys.readouterr()

    assert download(dest) == 0

    (why,) = hub_reasons(capsys)
    assert why.endswith(reason)
    assert tree(dest) == before


def test_download_restores_an_original_folder_that_holds_no_file(hub, tmp_path, capsys):
    manifest = json.loads(hub.files()[f"manifests/{VARIANT}.json"])
    manifest["original_folders"].append("empty")
    hub.files()[f"manifests/{VARIANT}.json"] = json.dumps(manifest).encode()
    dest = tmp_path / "deployment"
    assert download(dest) == 0
    (dest / "weights/empty").rmdir()
    capsys.readouterr()

    assert download(dest) == 0

    assert hub_reasons(capsys) == [f"the folder {dest / 'weights/empty'} is missing"]
    assert (dest / "weights/empty").is_dir()


def test_download_ignores_the_record_of_another_repo_or_variant(
    hub, cache_dir, tmp_path, capsys
):
    assert main(upload_argv(cache_dir, SIBLING)) == 0
    dest = tmp_path / "deployment"
    assert download(dest) == 0
    record = dest / RECORDS / f"{VARIANT}.json"
    copied = shutil.copyfile(record, record.with_name(f"{SIBLING}.json"))
    hub.fetched.clear()
    capsys.readouterr()

    assert download(dest, repo="example-org/another-tt-weights") == 0
    assert json.loads(record.read_text())["repo"] == "example-org/another-tt-weights"
    assert download(dest, SIBLING) == 0

    recorded = f"records {VARIANT} of {WEIGHTS_REPO}"
    assert hub_reasons(capsys) == [f"{record} {recorded}", f"{copied} {recorded}"]
    assert set(hub.fetched) == set(published(SIBLING)) - set(published(VARIANT))
    assert json.loads(copied.read_text())["variant"] == SIBLING


def test_download_that_replaces_tensors_removes_the_records(
    hub, cache_dir, tmp_path, capsys
):
    assert main(upload_argv(cache_dir, SIBLING)) == 0
    dest = tmp_path / "deployment"
    assert download(dest) == 0
    shared = f"tt_dit_cache/{SHARED}/embed.weight.tensorbin"
    hub.files()[shared] = hub.files()[shared][::-1]
    capsys.readouterr()

    assert download(dest, SIBLING) == 0
    assert os.listdir(dest / RECORDS) == [f"{SIBLING}.json"]
    assert download(dest) == 0

    cache = dest / "tt_dit_cache"
    no_record = "has no record of a complete download of"
    assert hub_reasons(capsys) == [
        f"{cache} {no_record} {SIBLING}",
        f"{cache} {no_record} {VARIANT}",
    ]
    assert (dest / shared).read_bytes() == hub.files()[shared]


def test_download_that_fills_a_removed_directory_removes_the_records(
    hub, cache_dir, tmp_path, capsys
):
    (tag,) = hub.tags
    assert main(upload_argv(cache_dir, SIBLING)) == 0
    shared = f"tt_dit_cache/{SHARED}/embed.weight.tensorbin"
    hub.files()[shared] = hub.files(tag)[shared][::-1]
    dest = tmp_path / "deployment"
    assert download(dest, revision=tag) == 0
    shutil.rmtree(dest / "tt_dit_cache" / SHARED)
    capsys.readouterr()

    assert download(dest, SIBLING) == 0
    assert os.listdir(dest / RECORDS) == [f"{SIBLING}.json"]
    assert (dest / shared).read_bytes() == hub.files()[shared]
    assert download(dest, revision=tag) == 0

    cache = dest / "tt_dit_cache"
    no_record = "has no record of a complete download of"
    assert hub_reasons(capsys) == [
        f"{cache} {no_record} {SIBLING}",
        f"{cache} {no_record} {VARIANT}",
    ]
    assert (dest / shared).read_bytes() == hub.files(tag)[shared]


WEIGHTS = {"repo": WEIGHTS_REPO, "variant": VARIANT, "tag": "t", "commit": "1" * 40}
RELEASE_OPTIONS = {
    "--image": KC_IMAGE,
    "--image-digest": "sha256:2222",
    "--tt-metal-sha": "aaa",
    "--server-sha": "bbb",
    "--model": "example-model",
    "--device": "galaxy",
}


def write_release(tmp_path, options=(), weights=WEIGHTS, kernel=KERNEL_MANIFEST):
    """The release file written for the fixtures; options replace or add option values."""
    kernel_manifest = tmp_path / "kernel_cache_manifest.json"
    kernel_manifest.write_text(json.dumps(kernel))
    out = tmp_path / "release.json"
    argv = ["release-file", "--base-image", BASE_IMAGE, "--out", str(out)]
    argv += ["--kernel-manifest", str(kernel_manifest)]
    if weights:
        weights_manifest = tmp_path / "tt_weights_manifest.json"
        weights_manifest.write_text(json.dumps({"schema_version": 1, **weights}))
        argv += ["--weights-manifest", str(weights_manifest)]
    for name, value in {**RELEASE_OPTIONS, **dict(options)}.items():
        argv += [name, value]
    assert main(argv) == 0
    return json.loads(out.read_text())


def test_release_file_names_image_kernel_cache_and_weights(tmp_path):
    release = write_release(tmp_path)

    assert release["tt_weights"] == WEIGHTS
    image = {"reference": KC_IMAGE, "digest": "sha256:2222", "base_image": BASE_IMAGE}
    assert release["image"] == image
    kernel_cache_env = {"TT_METAL_CACHE": KERNEL_CACHE_CONTAINER_DIR, **SET_VARIABLES}
    assert release["kernel_cache"]["env"] == kernel_cache_env
    assert write_release(tmp_path, weights=None)["tt_weights"] is None


WATCHED = {**KERNEL_MANIFEST["build_key_env"], "TT_METAL_WATCHER": "10"}
RELEASE_CHANGES = {
    "run-url": ({"options": {"--run-url": "https://ci.example/run/2"}}, True),
    "warm-ready-seconds": ({"options": {"--warm-ready-seconds": "301.5"}}, True),
    "image-reference": ({"options": {"--image": f"{KC_IMAGE}-again"}}, True),
    "weights-tag": ({"weights": {**WEIGHTS, "tag": "u"}}, True),
    "kernel-cache-size": ({"kernel": {**KERNEL_MANIFEST, "files": 4}}, True),
    "image-digest": ({"options": {"--image-digest": "sha256:3333"}}, False),
    "tt-metal-sha": ({"options": {"--tt-metal-sha": "ccc"}}, False),
    "server-sha": ({"options": {"--server-sha": "ccc"}}, False),
    "model": ({"options": {"--model": "another-model"}}, False),
    "device": ({"options": {"--device": "galaxy_quad"}}, False),
    "weights-commit": ({"weights": {**WEIGHTS, "commit": "2" * 40}}, False),
    "no-weights": ({"weights": None}, False),
    "kernel-cache-variable": (
        {"kernel": {**KERNEL_MANIFEST, "build_key_env": WATCHED}},
        False,
    ),
}


@pytest.mark.parametrize(
    "change, same_hash", RELEASE_CHANGES.values(), ids=RELEASE_CHANGES.keys()
)
def test_release_hash_changes_only_with_what_is_deployed(
    tmp_path, monkeypatch, change, same_hash
):
    first = write_release(tmp_path)
    monkeypatch.setattr(prebuilt_artifacts, "utc_now", lambda: "2000-01-01T00:00:00Z")

    second = write_release(tmp_path, **change)

    assert second["created_utc"] != first["created_utc"]
    assert (second["release_hash"] == first["release_hash"]) is same_hash
