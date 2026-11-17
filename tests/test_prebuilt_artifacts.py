# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import fnmatch
import inspect
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prebuilt_artifacts import KERNEL_CACHE_DOCKERFILE, main
from workflows.kernel_cache import (
    KERNEL_CACHE_CONTAINER_DIR,
    KERNEL_CACHE_MANIFEST_CONTAINER_PATH,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_IMAGE = "registry.example/media-server:1.0-aaa-bbb-1"
KC_IMAGE = f"{BASE_IMAGE}-kc"
WEIGHTS_REPO = "example-org/model-tt-weights"
ORIGINAL_REPO = "example-org/original-model"
LORA_REPO = "example-org/turbo-lora"


class FakeDocker:
    """Stands in for subprocess.run: records docker commands and answers inspect calls."""

    def __init__(self, image_env=(), repo_digests=(), failing=None):
        self.image_env = list(image_env)
        self.repo_digests = list(repo_digests)
        self.failing = failing
        self.commands = []
        self.build_context = None

    def __call__(self, command, capture_output=False, text=False):
        self.commands.append(command)
        stdout = ""
        if command[1:3] == ["image", "inspect"]:
            template = command[command.index("--format") + 1]
            if "Config.Env" in template:
                stdout = json.dumps(self.image_env)
            elif "RepoDigests" in template:
                stdout = json.dumps(self.repo_digests)
            else:
                stdout = "sha256:1111\n"
        elif command[1] == "build":
            context = Path(command[-1])
            self.build_context = {
                path.relative_to(context).as_posix(): path.read_bytes()
                for path in context.rglob("*")
                if path.is_file()
            }
        returncode = 1 if command[1] == self.failing else 0
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="boom")


class FakeHub:
    """In-memory Hugging Face Hub with the HfApi methods the CLI calls."""

    def __init__(self, repos=None):
        self.repos = {repo: dict(files) for repo, files in (repos or {}).items()}
        self.private = {}
        self.tags = {}
        self.head = {repo: self._commit() for repo in self.repos}
        self.calls = []

    def _commit(self):
        self._commits = getattr(self, "_commits", 0) + 1
        return f"{self._commits:040x}"

    def _record(self, method, **kwargs):
        self.calls.append((method, kwargs))

    def called(self, method):
        return [kwargs for name, kwargs in self.calls if name == method]

    @staticmethod
    def _matches(path, patterns):
        return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)

    def repo_exists(self, **kwargs):
        self._record("repo_exists", **kwargs)
        return kwargs["repo_id"] in self.repos

    def list_repo_refs(self, **kwargs):
        self._record("list_repo_refs", **kwargs)
        tags = self.tags.get(kwargs["repo_id"], {})
        return SimpleNamespace(tags=[SimpleNamespace(name=name) for name in tags])

    def create_repo(self, **kwargs):
        self._record("create_repo", **kwargs)
        repo = kwargs["repo_id"]
        self.repos[repo] = {}
        self.private[repo] = kwargs.get("private")
        self.head[repo] = self._commit()

    def create_branch(self, **kwargs):
        self._record("create_branch", **kwargs)

    def upload_folder(self, **kwargs):
        self._record("upload_folder", **kwargs)
        repo, folder = kwargs["repo_id"], Path(kwargs["folder_path"])
        prefix = kwargs["path_in_repo"] + "/"
        uploaded = {
            prefix + path.relative_to(folder).as_posix(): path.read_bytes()
            for path in folder.rglob("*")
            if path.is_file()
            and not self._matches(
                path.relative_to(folder).as_posix(), kwargs["ignore_patterns"]
            )
        }
        for path in list(self.repos[repo]):
            if path.startswith(prefix) and self._matches(
                path[len(prefix) :], kwargs["delete_patterns"]
            ):
                del self.repos[repo][path]
        self.repos[repo].update(uploaded)
        self.head[repo] = self._commit()
        return SimpleNamespace(oid=self.head[repo])

    def upload_file(self, **kwargs):
        self._record("upload_file", **kwargs)
        repo = kwargs["repo_id"]
        self.repos[repo][kwargs["path_in_repo"]] = kwargs["path_or_fileobj"]
        self.head[repo] = self._commit()
        return SimpleNamespace(oid=self.head[repo])

    def create_tag(self, **kwargs):
        self._record("create_tag", **kwargs)
        self.tags.setdefault(kwargs["repo_id"], {})[kwargs["tag"]] = kwargs["revision"]

    def model_info(self, **kwargs):
        self._record("model_info", **kwargs)
        return SimpleNamespace(sha=self.head[kwargs["repo_id"]])

    def hf_hub_download(self, **kwargs):
        self._record("hf_hub_download", **kwargs)
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.repos[kwargs["repo_id"]][kwargs["filename"]])
        return str(target)

    def snapshot_download(self, **kwargs):
        self._record("snapshot_download", **kwargs)
        for path, content in self.repos[kwargs["repo_id"]].items():
            if not self._matches(path, kwargs["allow_patterns"]):
                continue
            if self._matches(path, kwargs.get("ignore_patterns") or []):
                continue
            target = Path(kwargs["local_dir"]) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return kwargs["local_dir"]


def write_files(root, files):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def tree(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def kernel_cache(tmp_path):
    """A captured cache: two kernels and firmware under one build key, plus the shared PCH."""
    cache_dir = tmp_path / "kernel_cache"
    write_files(
        cache_dir,
        {
            "tt-metal-cache4242/kernels/reader/111/brisc/brisc.elf": b"elf-1",
            "tt-metal-cache4242/kernels/reader/111/brisc/brisc.elf.log": b"log",
            "tt-metal-cache4242/kernels/reader/111/.build_state": b"state",
            "tt-metal-cache4242/kernels/writer/222/ncrisc/ncrisc.o": b"object",
            "tt-metal-cache4242/kernels/writer/222/ncrisc/ncrisc.9001_77.o": b"temp",
            "tt-metal-cache4242/kernels/writer/222/.build_state.9001_77": b"temp",
            "tt-metal-cache4242/firmware/brisc/brisc.elf": b"firmware",
            "tt-metal-cache/pch/00ff/umbrella.hpp.gch": b"pch",
        },
    )
    return cache_dir


IMAGE_FILES = {
    "tt-metal-cache4242/kernels/reader/111/brisc/brisc.elf": b"elf-1",
    "tt-metal-cache4242/kernels/reader/111/.build_state": b"state",
    "tt-metal-cache4242/kernels/writer/222/ncrisc/ncrisc.o": b"object",
    "tt-metal-cache4242/firmware/brisc/brisc.elf": b"firmware",
    "tt-metal-cache/pch/00ff/umbrella.hpp.gch": b"pch",
}

BUILD_KEY_ENV = {
    "TT_METAL_INSPECTOR": "0",
    "TT_METAL_RISCV_DEBUG_INFO": None,
    "TT_MM_THROTTLE_PERF": "5",
    "MINIMAX_H3_TOPOLOGY": "linear",
    "MINIMAX_H3_VAE_OUTPUT": "yuv420",
    "TT_METAL_WATCHER": None,
}


def write_kernel_manifest(path, base_image=BASE_IMAGE):
    path.write_text(
        json.dumps(
            {
                "base_image": base_image,
                "kernel_cache_path": KERNEL_CACHE_CONTAINER_DIR,
                "build_keys": ["tt-metal-cache4242"],
                "kernel_hash_dirs": 2,
                "files": 5,
                "bytes": 27,
                "compiled_kernels_in_log": 2,
                "build_key_env": BUILD_KEY_ENV,
            }
        )
    )
    return path


def write_server_log(path, compiled):
    lines = ["2026-01-01 | info | Device | opening mesh"]
    lines += [
        f"2026-01-01 | info | BuildKernels | compiled kernel_{i}/{1000 + i}/ in {i} ms"
        for i in range(compiled)
    ]
    lines.append("2026-01-01 | info | Server | ready")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_kernel_cache_manifest_describes_cache_and_capture_env(
    kernel_cache, tmp_path, capsys
):
    docker = FakeDocker(
        image_env=["PATH=/usr/bin", "TT_MM_THROTTLE_PERF=5", "TT_METAL_INSPECTOR=1"]
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# secrets\nHF_TOKEN=hf_secret_value\nTT_METAL_INSPECTOR=0\nJWT_SECRET\n"
    )
    spec_json = tmp_path / "runtime_model_spec.json"
    spec_json.write_text(
        json.dumps(
            {
                "runtime_model_spec": {
                    "env_vars": {
                        "MINIMAX_H3_TOPOLOGY": "linear",
                        "MINIMAX_H3_VAE_OUTPUT": "yuv420",
                        "TT_METAL_WATCHER": "",
                    }
                },
                "runtime_config": {},
            }
        )
    )
    out = tmp_path / "out" / "kernel_cache_manifest.json"

    exit_code = main(
        [
            "kernel-cache",
            "manifest",
            "--cache-dir",
            str(kernel_cache),
            "--server-log",
            str(write_server_log(tmp_path / "server.log", compiled=2)),
            "--base-image",
            BASE_IMAGE,
            "--env-file",
            str(env_file),
            "--spec-json",
            str(spec_json),
            "--out",
            str(out),
        ],
        docker_run=docker,
    )

    assert exit_code == 0
    manifest = json.loads(out.read_text())
    assert manifest["base_image"] == BASE_IMAGE
    assert manifest["kernel_cache_path"] == KERNEL_CACHE_CONTAINER_DIR
    assert manifest["build_keys"] == ["tt-metal-cache4242"]
    assert manifest["kernel_hash_dirs"] == 2
    assert manifest["files"] == len(IMAGE_FILES)
    assert manifest["bytes"] == sum(len(data) for data in IMAGE_FILES.values())
    assert manifest["compiled_kernels_in_log"] == 2
    assert manifest["build_key_env"] == BUILD_KEY_ENV
    assert docker.commands == [
        ["docker", "image", "inspect", "--format", "{{json .Config.Env}}", BASE_IMAGE]
    ]
    captured = capsys.readouterr()
    assert len(captured.out.splitlines()) == 4
    assert "hf_secret_value" not in captured.out + captured.err + out.read_text()


def test_kernel_cache_manifest_fails_when_nothing_was_captured(tmp_path, capsys):
    empty = tmp_path / "kernel_cache"
    empty.mkdir()
    out = tmp_path / "manifest.json"

    exit_code = main(
        [
            "kernel-cache",
            "manifest",
            "--cache-dir",
            str(empty),
            "--server-log",
            str(write_server_log(tmp_path / "server.log", compiled=0)),
            "--base-image",
            BASE_IMAGE,
            "--out",
            str(out),
        ],
        docker_run=FakeDocker(),
    )

    assert exit_code == 1
    assert "nothing was captured" in capsys.readouterr().err
    assert not out.exists()


def test_kernel_cache_build_image_stages_context_builds_and_pushes(
    kernel_cache, tmp_path, capsys
):
    manifest = write_kernel_manifest(tmp_path / "manifest.json")
    digest = f"{BASE_IMAGE.rsplit(':', 1)[0]}@sha256:2222"
    docker = FakeDocker(repo_digests=["registry.example/other@sha256:9999", digest])

    exit_code = main(
        [
            "kernel-cache",
            "build-image",
            "--base-image",
            BASE_IMAGE,
            "--cache-dir",
            str(kernel_cache),
            "--manifest",
            str(manifest),
            "--tag",
            KC_IMAGE,
            "--push",
        ],
        docker_run=docker,
    )

    assert exit_code == 0
    build, inspect_id, push, inspect_digests = docker.commands
    context = Path(build[-1])
    assert build == [
        "docker",
        "build",
        "--build-arg",
        f"BASE_IMAGE={BASE_IMAGE}",
        "--tag",
        KC_IMAGE,
        "--file",
        str(context / "Dockerfile"),
        str(context),
    ]
    assert push == ["docker", "push", KC_IMAGE]
    assert not context.exists()

    staged = dict(docker.build_context)
    dockerfile = staged.pop("Dockerfile").decode()
    assert staged.pop("kernel_cache_manifest.json") == manifest.read_bytes()
    assert staged == {
        f"kernel_cache/{path}": data for path, data in IMAGE_FILES.items()
    }
    assert dockerfile.startswith(KERNEL_CACHE_DOCKERFILE.read_text())
    assert [line for line in dockerfile.splitlines() if line.startswith("ENV ")] == [
        f"ENV TT_METAL_CACHE={KERNEL_CACHE_CONTAINER_DIR}",
        'ENV TT_METAL_INSPECTOR="0"',
        'ENV TT_MM_THROTTLE_PERF="5"',
        'ENV MINIMAX_H3_TOPOLOGY="linear"',
        'ENV MINIMAX_H3_VAE_OUTPUT="yuv420"',
    ]

    output = capsys.readouterr().out.splitlines()
    assert "image-id: sha256:1111" in output
    assert f"digest: {digest}" in output


def test_kernel_cache_build_image_without_push_stays_local(
    kernel_cache, tmp_path, capsys
):
    manifest = write_kernel_manifest(tmp_path / "manifest.json")
    docker = FakeDocker()

    exit_code = main(
        [
            "kernel-cache",
            "build-image",
            "--base-image",
            BASE_IMAGE,
            "--cache-dir",
            str(kernel_cache),
            "--manifest",
            str(manifest),
            "--tag",
            KC_IMAGE,
        ],
        docker_run=docker,
    )

    assert exit_code == 0
    assert [command[1] for command in docker.commands] == ["build", "image"]
    assert "digest:" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "base_image, failing, message",
    [
        ("registry.example/media-server:other", None, "made for base image"),
        (BASE_IMAGE, "build", "docker build failed"),
        (BASE_IMAGE, "push", "docker push"),
    ],
)
def test_kernel_cache_build_image_failures_exit_non_zero(
    kernel_cache, tmp_path, capsys, base_image, failing, message
):
    manifest = write_kernel_manifest(tmp_path / "manifest.json")

    exit_code = main(
        [
            "kernel-cache",
            "build-image",
            "--base-image",
            base_image,
            "--cache-dir",
            str(kernel_cache),
            "--manifest",
            str(manifest),
            "--tag",
            KC_IMAGE,
            "--push",
        ],
        docker_run=FakeDocker(failing=failing),
    )

    assert exit_code == 1
    assert message in capsys.readouterr().err


def test_kernel_cache_dockerfile_keeps_base_image_runtime_contract():
    instructions = [
        line.split()
        for line in KERNEL_CACHE_DOCKERFILE.read_text().splitlines()
        if line and not line.startswith("#")
    ]

    assert instructions == [
        ["ARG", "BASE_IMAGE"],
        ["FROM", "${BASE_IMAGE}"],
        ["COPY", "--chmod=777", "kernel_cache/", f"{KERNEL_CACHE_CONTAINER_DIR}/"],
        ["RUN", "chmod", "777", KERNEL_CACHE_CONTAINER_DIR],
        [
            "COPY",
            "--chmod=644",
            "kernel_cache_manifest.json",
            KERNEL_CACHE_MANIFEST_CONTAINER_PATH,
        ],
        ["ENV", f"TT_METAL_CACHE={KERNEL_CACHE_CONTAINER_DIR}"],
    ]


def test_check_log_passes_when_nothing_was_compiled(tmp_path, capsys):
    server_log = write_server_log(tmp_path / "server.log", compiled=0)

    exit_code = main(["kernel-cache", "check-log", "--server-log", str(server_log)])

    assert exit_code == 0
    assert capsys.readouterr().out == "compiled kernels: 0 (allowed: 0)\n"


def test_check_log_fails_and_names_the_first_20_compiled_kernels(tmp_path, capsys):
    server_log = write_server_log(tmp_path / "server.log", compiled=25)
    argv = ["kernel-cache", "check-log", "--server-log", str(server_log)]

    assert main(argv) == 1
    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith("compiled kernels: 25")
    assert output[1:] == [f"kernel_{i}/{1000 + i}" for i in range(20)]

    assert main(argv + ["--max-compiled", "25"]) == 0


def test_cli_runs_as_a_script_from_the_repo_root(tmp_path):
    server_log = write_server_log(tmp_path / "server.log", compiled=1)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/prebuilt_artifacts.py",
            "kernel-cache",
            "check-log",
            "--server-log",
            str(server_log),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "kernel_0/1000" in result.stdout


@pytest.fixture
def tt_dit_cache(tmp_path):
    """Two finished conversions, each with its (empty) completion marker."""
    cache_dir = tmp_path / "volume" / "tt_dit_cache"
    write_files(
        cache_dir,
        {
            "model/transformer_lora/TP4_mesh4x8_bf16/blocks.0.weight.tensorbin": b"abc",
            "model/transformer_lora/TP4_mesh4x8_bf16/blocks.0.bias.tensorbin": b"defgh",
            "model/transformer_lora/TP4_mesh4x8_bf16/cache_dict.json": b"",
            "model/text_encoder/mesh4x8_bf16/embed.weight.tensorbin": b"ijklmno",
            "model/text_encoder/mesh4x8_bf16/cache_dict.json": b"",
        },
    )
    return cache_dir


TRANSFORMER_DIR = "model/transformer_lora/TP4_mesh4x8_bf16"
TEXT_ENCODER_DIR = "model/text_encoder/mesh4x8_bf16"
TENSORS = {
    f"{TRANSFORMER_DIR}/blocks.0.weight.tensorbin": b"abc",
    f"{TRANSFORMER_DIR}/blocks.0.bias.tensorbin": b"defgh",
    f"{TEXT_ENCODER_DIR}/embed.weight.tensorbin": b"ijklmno",
}


def upload_argv(cache_dir, manifest_out, tag="ttm-aaa-srv-bbb"):
    return [
        "tt-weights",
        "upload",
        "--cache-dir",
        str(cache_dir),
        "--repo",
        WEIGHTS_REPO,
        "--revision",
        "main",
        "--tag",
        tag,
        "--meta",
        "tt_metal_sha=aaa",
        "server_sha=bbb",
        "--manifest-out",
        str(manifest_out),
    ]


def download_argv(dest, revision="main"):
    return [
        "tt-weights",
        "download",
        "--repo",
        WEIGHTS_REPO,
        "--revision",
        revision,
        "--dest",
        str(dest),
    ]


def test_tt_weights_upload_creates_private_repo_manifest_and_tag(
    tt_dit_cache, tmp_path
):
    hub = FakeHub()
    manifest_out = tmp_path / "tt_weights_manifest.json"

    exit_code = main(upload_argv(tt_dit_cache, manifest_out), hf_api=hub)

    assert exit_code == 0
    assert hub.private[WEIGHTS_REPO] is True
    remote = dict(hub.repos[WEIGHTS_REPO])
    remote_manifest = json.loads(remote.pop("tt_weights_manifest.json"))
    assert remote == {f"tt_dit_cache/{path}": data for path, data in TENSORS.items()}

    manifest = json.loads(manifest_out.read_text())
    assert manifest.pop("commit") == hub.head[WEIGHTS_REPO]
    assert manifest == remote_manifest
    assert manifest["directories"] == [
        {"path": TEXT_ENCODER_DIR, "files": 1, "bytes": 7},
        {"path": TRANSFORMER_DIR, "files": 2, "bytes": 8},
    ]
    assert (manifest["files"], manifest["bytes"]) == (3, 15)
    assert manifest["meta"] == {"tt_metal_sha": "aaa", "server_sha": "bbb"}
    assert manifest["revision"] == "main"
    assert manifest["created_utc"].endswith("Z")
    assert hub.tags[WEIGHTS_REPO] == {"ttm-aaa-srv-bbb": hub.head[WEIGHTS_REPO]}
    assert hub.called("upload_folder")[0]["revision"] == "main"


@pytest.mark.parametrize(
    "broken_file, removed_file, reason",
    [
        (None, f"{TEXT_ENCODER_DIR}/cache_dict.json", "no cache_dict.json"),
        (f"{TRANSFORMER_DIR}/x.tensorbin.tmp.4242.0", None, "1 temp file(s)"),
    ],
)
def test_tt_weights_upload_refuses_an_unfinished_conversion(
    tt_dit_cache, tmp_path, capsys, broken_file, removed_file, reason
):
    if broken_file:
        (tt_dit_cache / broken_file).write_bytes(b"partial")
    if removed_file:
        (tt_dit_cache / removed_file).unlink()
    hub = FakeHub()
    manifest_out = tmp_path / "tt_weights_manifest.json"

    exit_code = main(upload_argv(tt_dit_cache, manifest_out), hf_api=hub)

    assert exit_code == 1
    assert reason in capsys.readouterr().err
    assert hub.calls == []
    assert not manifest_out.exists()


def test_tt_weights_upload_mirrors_into_an_existing_repo(tt_dit_cache, tmp_path):
    stale = f"tt_dit_cache/{TRANSFORMER_DIR}/renamed.tensorbin"
    hub = FakeHub({WEIGHTS_REPO: {stale: b"old", "README.md": b"notes"}})
    hub.tags[WEIGHTS_REPO] = {"ttm-old-srv-old": hub.head[WEIGHTS_REPO]}

    exit_code = main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub)

    assert exit_code == 0
    assert hub.called("create_repo") == []
    assert stale not in hub.repos[WEIGHTS_REPO]
    assert hub.repos[WEIGHTS_REPO]["README.md"] == b"notes"
    assert set(hub.tags[WEIGHTS_REPO]) == {"ttm-old-srv-old", "ttm-aaa-srv-bbb"}


def test_tt_weights_upload_refuses_to_move_an_existing_tag(
    tt_dit_cache, tmp_path, capsys
):
    hub = FakeHub({WEIGHTS_REPO: {}})
    hub.tags[WEIGHTS_REPO] = {"ttm-aaa-srv-bbb": hub.head[WEIGHTS_REPO]}

    exit_code = main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub)

    assert exit_code == 1
    assert "tag ttm-aaa-srv-bbb already exists" in capsys.readouterr().err
    assert hub.called("upload_folder") == []
    assert hub.repos[WEIGHTS_REPO] == {}


def test_tt_weights_download_verifies_then_marks_directories_complete(
    tt_dit_cache, tmp_path
):
    hub = FakeHub()
    assert main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub) == 0
    dest = tmp_path / "deployment" / "tt_dit_cache"

    exit_code = main(download_argv(dest, revision="ttm-aaa-srv-bbb"), hf_api=hub)

    assert exit_code == 0
    assert tree(dest) == tree(tt_dit_cache)
    commit = hub.head[WEIGHTS_REPO]
    assert hub.called("model_info")[0]["revision"] == "ttm-aaa-srv-bbb"
    assert hub.called("hf_hub_download")[0]["revision"] == commit
    assert hub.called("snapshot_download")[0]["revision"] == commit


def test_tt_weights_download_leaves_no_marker_when_the_manifest_does_not_match(
    tt_dit_cache, tmp_path, capsys
):
    hub = FakeHub()
    assert main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub) == 0
    del hub.repos[WEIGHTS_REPO][
        f"tt_dit_cache/{TRANSFORMER_DIR}/blocks.0.bias.tensorbin"
    ]
    dest = tmp_path / "deployment" / "tt_dit_cache"
    write_files(dest, {f"{TRANSFORMER_DIR}/cache_dict.json": b""})

    exit_code = main(download_argv(dest), hf_api=hub)

    assert exit_code == 1
    assert f"{TRANSFORMER_DIR}: 1 files, 3 bytes" in capsys.readouterr().err
    assert not list(dest.rglob("cache_dict.json"))
    assert (dest / TEXT_ENCODER_DIR / "embed.weight.tensorbin").read_bytes() == (
        b"ijklmno"
    )


ORIGINAL_FILES = {
    "model_index.json": b"{}",
    "scheduler/scheduler_config.json": b"{}",
    "tokenizer/tokenizer.json": b"{}",
    "transformer/config.json": b"{}",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"big",
    "transformer_ref/config.json": b"{}",
    "transformer_ref/diffusion_pytorch_model-00001-of-00002.safetensors": b"big",
    "text_encoder/config.json": b"{}",
    "text_encoder/preprocessor_config.json": b"{}",
    "text_encoder/model.safetensors.index.json": json.dumps(
        {
            "weight_map": {
                "model.language_model.layers.0.weight": "model-00001-of-00003.safetensors",
                "model.visual.blocks.0.weight": "model-00002-of-00003.safetensors",
                "model.visual.blocks.1.weight": "model-00002-of-00003.safetensors",
                "lm_head.weight": "model-00003-of-00003.safetensors",
            }
        }
    ).encode(),
    "text_encoder/model-00001-of-00003.safetensors": b"big",
    "text_encoder/model-00002-of-00003.safetensors": b"vision",
    "text_encoder/model-00003-of-00003.safetensors": b"big",
    "vae/config.json": b"{}",
    "vae/diffusion_pytorch_model.safetensors": b"vae",
    "audio_vae/config.json": b"{}",
    "audio_vae/diffusion_pytorch_model.safetensors": b"big",
    "FL2VA/original_bundle.safetensors": b"big",
    "assets/sample.mp4": b"big",
}


@pytest.mark.parametrize(
    "variant, transformer, other_transformer",
    [
        ("fl2va", "transformer", "transformer_ref"),
        ("ref2va", "transformer_ref", "transformer"),
    ],
)
def test_original_subset_downloads_only_what_a_cached_start_reads(
    tmp_path, variant, transformer, other_transformer
):
    hub = FakeHub(
        {ORIGINAL_REPO: ORIGINAL_FILES, LORA_REPO: {"turbo.safetensors": b"lora"}}
    )
    dest = tmp_path / "weights"

    exit_code = main(
        [
            "tt-weights",
            "original-subset",
            "--repo",
            ORIGINAL_REPO,
            "--variant",
            variant,
            "--dest",
            str(dest),
            "--lora",
            f"{LORA_REPO}:turbo.safetensors",
        ],
        hf_api=hub,
    )

    assert exit_code == 0
    assert set(tree(dest)) == {
        "model_index.json",
        "scheduler/scheduler_config.json",
        "tokenizer/tokenizer.json",
        f"{transformer}/config.json",
        "text_encoder/config.json",
        "text_encoder/preprocessor_config.json",
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-00002-of-00003.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "audio_vae/config.json",
        "lightx2v/turbo.safetensors",
    }
    assert not (dest / other_transformer).exists()


def test_original_subset_creates_the_folders_the_pipeline_requires(tmp_path):
    files = {
        path: data
        for path, data in ORIGINAL_FILES.items()
        if not path.startswith(("audio_vae/", "transformer/"))
    }
    hub = FakeHub({ORIGINAL_REPO: files})
    dest = tmp_path / "weights"

    exit_code = main(
        [
            "tt-weights",
            "original-subset",
            "--repo",
            ORIGINAL_REPO,
            "--variant",
            "fl2va",
            "--dest",
            str(dest),
        ],
        hf_api=hub,
    )

    assert exit_code == 0
    for folder in ("transformer", "text_encoder", "vae", "audio_vae"):
        assert (dest / folder).is_dir()
    assert not (dest / "lightx2v").exists()


def test_tt_weights_calls_match_the_installed_huggingface_hub(tt_dit_cache, tmp_path):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    hub = FakeHub(
        {ORIGINAL_REPO: ORIGINAL_FILES, LORA_REPO: {"turbo.safetensors": b"lora"}}
    )
    assert main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub) == 0
    assert (
        main(upload_argv(tt_dit_cache, tmp_path / "m.json", "tag-2"), hf_api=hub) == 0
    )
    assert main(download_argv(tmp_path / "dest"), hf_api=hub) == 0
    subset_argv = ["tt-weights", "original-subset", "--repo", ORIGINAL_REPO]
    subset_argv += ["--variant", "fl2va", "--dest", str(tmp_path / "weights")]
    subset_argv += ["--lora", f"{LORA_REPO}:turbo.safetensors"]
    assert main(subset_argv, hf_api=hub) == 0

    api = huggingface_hub.HfApi()
    for method, kwargs in hub.calls:
        inspect.signature(getattr(api, method)).bind(**kwargs)


def test_tt_weights_needs_the_token_in_the_environment(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    exit_code = main(download_argv(tmp_path / "dest"))

    assert exit_code == 1
    assert capsys.readouterr().err == "error: HF_TOKEN is not set\n"
    assert not (tmp_path / "dest").exists()


def release_argv(tmp_path, out, weights_manifest=None, digest="sha256:2222"):
    argv = [
        "release-hash",
        "--base-image",
        BASE_IMAGE,
        "--image",
        KC_IMAGE,
        "--image-digest",
        digest,
        "--kernel-manifest",
        str(write_kernel_manifest(tmp_path / "kernel_cache_manifest.json")),
        "--tt-metal-sha",
        "aaa",
        "--server-sha",
        "bbb",
        "--model",
        "example-model",
        "--device",
        "galaxy",
        "--out",
        str(out),
    ]
    if weights_manifest:
        argv += ["--weights-manifest", str(weights_manifest)]
    return argv


def test_release_hash_names_image_kernel_cache_and_weights(tt_dit_cache, tmp_path):
    weights_manifest = tmp_path / "tt_weights_manifest.json"
    hub = FakeHub()
    assert main(upload_argv(tt_dit_cache, weights_manifest), hf_api=hub) == 0
    out = tmp_path / "release.json"
    extra = ["--warm-ready-seconds", "731.5", "--run-url", "https://ci.example/run/1"]

    exit_code = main(release_argv(tmp_path, out, weights_manifest) + extra)

    assert exit_code == 0
    release = json.loads(out.read_text())
    assert len(release["release_hash"]) == 64
    assert release["image"] == {
        "reference": KC_IMAGE,
        "digest": "sha256:2222",
        "base_image": BASE_IMAGE,
    }
    assert release["kernel_cache"]["path"] == KERNEL_CACHE_CONTAINER_DIR
    assert release["kernel_cache"]["env"] == {
        "TT_METAL_CACHE": KERNEL_CACHE_CONTAINER_DIR,
        "TT_METAL_INSPECTOR": "0",
        "TT_MM_THROTTLE_PERF": "5",
        "MINIMAX_H3_TOPOLOGY": "linear",
        "MINIMAX_H3_VAE_OUTPUT": "yuv420",
    }
    assert release["kernel_cache"]["kernel_hash_dirs"] == 2
    assert release["tt_weights"] == {
        "repo": WEIGHTS_REPO,
        "revision": "main",
        "tag": "ttm-aaa-srv-bbb",
        "commit": hub.head[WEIGHTS_REPO],
        "files": 3,
        "bytes": 15,
        "meta": {"tt_metal_sha": "aaa", "server_sha": "bbb"},
    }
    assert (release["tt_metal_sha"], release["server_sha"]) == ("aaa", "bbb")
    assert (release["model"], release["device"]) == ("example-model", "galaxy")
    assert release["warm_ready_seconds"] == 731.5
    assert release["run_url"] == "https://ci.example/run/1"


def test_release_hash_depends_only_on_what_is_deployed(tmp_path):
    def release(name, **kwargs):
        out = tmp_path / name
        assert main(release_argv(tmp_path, out, **kwargs)) == 0
        return json.loads(out.read_text())

    first = release("first.json")
    again = release("again.json")
    other_image = release("other.json", digest="sha256:3333")

    assert first["tt_weights"] is None
    assert first["warm_ready_seconds"] is None
    assert first["release_hash"] == again["release_hash"]
    assert first["release_hash"] != other_image["release_hash"]


def test_release_hash_refuses_a_kernel_manifest_of_another_image(tmp_path, capsys):
    out = tmp_path / "release.json"
    argv = release_argv(tmp_path, out)
    write_kernel_manifest(
        tmp_path / "kernel_cache_manifest.json", "registry.example/media-server:other"
    )

    assert main(argv) == 1
    assert "made for base image" in capsys.readouterr().err
    assert not out.exists()
