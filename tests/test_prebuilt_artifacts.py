# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import fnmatch
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
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
        self.environments = []
        self.build_context = None

    def __call__(self, command, capture_output=False, text=False, env=None):
        self.commands.append(command)
        self.environments.append(env)
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
    """In-memory Hugging Face Hub with the HfApi methods the CLI calls.

    Every commit is kept, so a tag or a commit hash still reads the files it
    pointed at. Like the real client on a large folder, upload_folder commits
    `batch` files at a time, with the deletions in its first commit; it loses the
    connection after `interrupt_after` commits.
    """

    def __init__(self, repos=None, batch=None):
        self.batch = batch
        self.interrupt_after = None
        self.private = {}
        self.branches = {}
        self.tags = {}
        self.commits = {}
        self.calls = []
        for repo, files in (repos or {}).items():
            self._create(repo, files, private=True)

    def _create(self, repo, files, private):
        self.private[repo] = private
        self.branches[repo] = {}
        self.tags[repo] = {}
        self._commit(repo, "main", files)

    def _commit(self, repo, branch, files):
        commit = f"{len(self.commits) + 1:040x}"
        self.commits[commit] = dict(files)
        self.branches[repo][branch] = commit
        return commit

    def _record(self, method, **kwargs):
        self.calls.append((method, kwargs))

    def called(self, method):
        return [kwargs for name, kwargs in self.calls if name == method]

    def commit(self, repo, revision="main"):
        for refs in (self.tags[repo], self.branches[repo]):
            if revision in refs:
                return refs[revision]
        return revision

    def files(self, repo, revision=None):
        return self.commits[self.commit(repo, revision or "main")]

    @staticmethod
    def _matches(path, patterns):
        return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)

    def repo_exists(self, **kwargs):
        self._record("repo_exists", **kwargs)
        return kwargs["repo_id"] in self.branches

    def model_info(self, **kwargs):
        self._record("model_info", **kwargs)
        repo = kwargs["repo_id"]
        return SimpleNamespace(
            sha=self.commit(repo, kwargs.get("revision") or "main"),
            private=self.private[repo],
        )

    def list_repo_refs(self, **kwargs):
        self._record("list_repo_refs", **kwargs)
        tags = self.tags[kwargs["repo_id"]]
        return SimpleNamespace(
            tags=[
                SimpleNamespace(name=name, target_commit=commit)
                for name, commit in tags.items()
            ]
        )

    def create_repo(self, **kwargs):
        self._record("create_repo", **kwargs)
        self._create(kwargs["repo_id"], {}, private=kwargs.get("private"))

    def create_branch(self, **kwargs):
        self._record("create_branch", **kwargs)
        branches = self.branches[kwargs["repo_id"]]
        branches.setdefault(kwargs["branch"], branches["main"])

    def file_exists(self, **kwargs):
        self._record("file_exists", **kwargs)
        return kwargs["filename"] in self.files(
            kwargs["repo_id"], kwargs.get("revision")
        )

    def delete_file(self, **kwargs):
        self._record("delete_file", **kwargs)
        repo, branch = kwargs["repo_id"], kwargs["revision"]
        files = dict(self.files(repo, branch))
        del files[kwargs["path_in_repo"]]
        return SimpleNamespace(oid=self._commit(repo, branch, files))

    def upload_folder(self, **kwargs):
        self._record("upload_folder", **kwargs)
        repo, branch = kwargs["repo_id"], kwargs["revision"]
        folder, prefix = Path(kwargs["folder_path"]), kwargs["path_in_repo"] + "/"
        uploads = sorted(
            (prefix + path.relative_to(folder).as_posix(), path.read_bytes())
            for path in folder.rglob("*")
            if path.is_file()
            and not self._matches(
                path.relative_to(folder).as_posix(), kwargs["ignore_patterns"]
            )
        )
        replaced = {path for path, _ in uploads}
        files = {
            path: data
            for path, data in self.files(repo, branch).items()
            if path in replaced
            or not path.startswith(prefix)
            or not self._matches(path[len(prefix) :], kwargs["delete_patterns"])
        }
        commit = self.commit(repo, branch)
        batch = self.batch or len(uploads)
        for number, first in enumerate(range(0, len(uploads), batch)):
            if number == self.interrupt_after:
                raise ConnectionError("connection lost")
            files.update(uploads[first : first + batch])
            commit = self._commit(repo, branch, files)
        return SimpleNamespace(oid=commit)

    def upload_file(self, **kwargs):
        self._record("upload_file", **kwargs)
        repo, branch = kwargs["repo_id"], kwargs["revision"]
        files = dict(self.files(repo, branch))
        files[kwargs["path_in_repo"]] = kwargs["path_or_fileobj"]
        return SimpleNamespace(oid=self._commit(repo, branch, files))

    def list_repo_tree(self, **kwargs):
        self._record("list_repo_tree", **kwargs)
        prefix = kwargs["path_in_repo"] + "/"
        files = {
            path: data
            for path, data in self.files(kwargs["repo_id"], kwargs["revision"]).items()
            if path.startswith(prefix)
        }
        folders = sorted({PurePosixPath(path).parent.as_posix() for path in files})
        return [SimpleNamespace(path=folder) for folder in folders] + [
            SimpleNamespace(path=path, size=len(data)) for path, data in files.items()
        ]

    def create_tag(self, **kwargs):
        self._record("create_tag", **kwargs)
        repo = kwargs["repo_id"]
        self.tags[repo][kwargs["tag"]] = self.commit(repo, kwargs["revision"])

    def hf_hub_download(self, **kwargs):
        self._record("hf_hub_download", **kwargs)
        files = self.files(kwargs["repo_id"], kwargs.get("revision"))
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[kwargs["filename"]])
        return str(target)

    def snapshot_download(self, **kwargs):
        self._record("snapshot_download", **kwargs)
        files = self.files(kwargs["repo_id"], kwargs.get("revision"))
        for path, content in files.items():
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


def manifest_argv(cache_dir, tmp_path, compiled=2, extra=()):
    return [
        "kernel-cache",
        "manifest",
        "--cache-dir",
        str(cache_dir),
        "--server-log",
        str(write_server_log(tmp_path / "server.log", compiled)),
        "--base-image",
        BASE_IMAGE,
        "--out",
        str(tmp_path / "out" / "kernel_cache_manifest.json"),
        *extra,
    ]


def build_image_argv(cache_dir, manifest, base_image=BASE_IMAGE, extra=()):
    return [
        "kernel-cache",
        "build-image",
        "--base-image",
        base_image,
        "--cache-dir",
        str(cache_dir),
        "--manifest",
        str(manifest),
        "--tag",
        KC_IMAGE,
        *extra,
    ]


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
    extra = ["--env-file", str(env_file), "--spec-json", str(spec_json)]

    exit_code = main(
        manifest_argv(kernel_cache, tmp_path, extra=extra), docker_run=docker
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


@pytest.mark.parametrize("wrapper", ["runtime_model_spec", "model_spec", None])
def test_kernel_cache_manifest_reads_every_spec_format_run_py_reads(
    kernel_cache, tmp_path, wrapper
):
    model_spec = {"env_vars": {"MINIMAX_H3_TOPOLOGY": "linear"}}
    spec_json = tmp_path / "spec.json"
    spec_json.write_text(
        json.dumps(
            {wrapper: model_spec, "runtime_config": {}} if wrapper else model_spec
        )
    )
    extra = ["--spec-json", str(spec_json)]

    exit_code = main(
        manifest_argv(kernel_cache, tmp_path, extra=extra), docker_run=FakeDocker()
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / "out" / "kernel_cache_manifest.json").read_text())
    assert manifest["build_key_env"]["MINIMAX_H3_TOPOLOGY"] == "linear"


@pytest.mark.parametrize(
    "option, content, message",
    [
        ("--spec-json", '{"runtime_config": {}}', "no env_vars in the model spec"),
        (
            "--env-file",
            "TT_METAL_INSPECTOR\n",
            "takes TT_METAL_INSPECTOR from the environment",
        ),
    ],
)
def test_kernel_cache_manifest_refuses_build_key_values_it_cannot_resolve(
    kernel_cache, tmp_path, capsys, option, content, message
):
    source = tmp_path / "source"
    source.write_text(content)

    exit_code = main(
        manifest_argv(kernel_cache, tmp_path, extra=[option, str(source)]),
        docker_run=FakeDocker(),
    )

    assert exit_code == 1
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_kernel_cache_manifest_says_which_env_sources_were_not_given(
    kernel_cache, tmp_path, capsys
):
    exit_code = main(manifest_argv(kernel_cache, tmp_path), docker_run=FakeDocker())

    assert exit_code == 0
    assert "env: no --env-file or --spec-json given" in capsys.readouterr().out


def test_kernel_cache_manifest_fails_when_nothing_was_captured(tmp_path, capsys):
    empty = tmp_path / "kernel_cache"
    empty.mkdir()

    exit_code = main(
        manifest_argv(empty, tmp_path, compiled=0), docker_run=FakeDocker()
    )

    assert exit_code == 1
    assert "nothing was captured" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_kernel_cache_manifest_refuses_a_log_without_the_compiles_of_the_capture(
    kernel_cache, tmp_path, capsys
):
    exit_code = main(
        manifest_argv(kernel_cache, tmp_path, compiled=0), docker_run=FakeDocker()
    )

    assert exit_code == 1
    assert "has no compile line although the cache holds 2 kernel hash folders" in (
        capsys.readouterr().err
    )
    assert not (tmp_path / "out").exists()


def test_kernel_cache_build_image_stages_context_builds_and_pushes(
    kernel_cache, tmp_path, capsys
):
    manifest = write_kernel_manifest(tmp_path / "manifest.json")
    digest = f"{BASE_IMAGE.rsplit(':', 1)[0]}@sha256:2222"
    docker = FakeDocker(repo_digests=["registry.example/other@sha256:9999", digest])

    exit_code = main(
        build_image_argv(kernel_cache, manifest, extra=["--push"]), docker_run=docker
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
    build_environment, *other_environments = docker.environments
    assert build_environment == {**os.environ, "DOCKER_BUILDKIT": "1"}
    assert other_environments == [None, None, None]
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

    exit_code = main(build_image_argv(kernel_cache, manifest), docker_run=docker)

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
        build_image_argv(kernel_cache, manifest, base_image, extra=["--push"]),
        docker_run=FakeDocker(failing=failing),
    )

    assert exit_code == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("link_to_folder", [False, True])
def test_kernel_cache_commands_refuse_symbolic_links(
    kernel_cache, tmp_path, capsys, link_to_folder
):
    write_files(tmp_path / "host", {"secrets/.env": b"JWT_SECRET=value"})
    target = tmp_path / "host" / "secrets"
    link = kernel_cache / "tt-metal-cache4242/kernels/reader/111/linked.o"
    link.symlink_to(target if link_to_folder else target / ".env")
    manifest = write_kernel_manifest(tmp_path / "manifest.json")
    docker = FakeDocker()

    assert main(build_image_argv(kernel_cache, manifest), docker_run=docker) == 1
    assert "symbolic links and special files are not packaged" in (
        capsys.readouterr().err
    )
    assert main(manifest_argv(kernel_cache, tmp_path), docker_run=docker) == 1
    assert "kernels/reader/111/linked.o" in capsys.readouterr().err
    assert docker.commands == []
    assert not (tmp_path / "out").exists()


def test_kernel_cache_build_image_refuses_a_directory_the_manifest_does_not_describe(
    kernel_cache, tmp_path, capsys
):
    manifest = write_kernel_manifest(tmp_path / "manifest.json")
    checkout = tmp_path / "checkout"
    write_files(checkout, {".env": b"JWT_SECRET=value"})
    docker = FakeDocker()

    assert main(build_image_argv(checkout, manifest), docker_run=docker) == 1
    assert "has no tt-metal-cache4242" in capsys.readouterr().err

    (kernel_cache / "tt-metal-cache4242/kernels/reader/111/added.o").write_bytes(b"x")
    assert main(build_image_argv(kernel_cache, manifest), docker_run=docker) == 1
    assert "holds 6 files, 28 bytes; the manifest says 5 files, 27 bytes" in (
        capsys.readouterr().err
    )
    assert docker.commands == []


def test_path_options_refuse_an_empty_value(tmp_path, capsys):
    manifest = write_kernel_manifest(tmp_path / "manifest.json")
    docker = FakeDocker()

    with pytest.raises(SystemExit) as stopped:
        main(build_image_argv("", manifest), docker_run=docker)

    assert stopped.value.code == 2
    assert "argument --cache-dir: the path is empty" in capsys.readouterr().err
    assert docker.commands == []


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


def test_check_log_fails_on_a_log_that_shows_no_server_start(tmp_path, capsys):
    empty = tmp_path / "empty.log"
    empty.write_text("")
    missing = tmp_path / "missing.log"

    assert main(["kernel-cache", "check-log", "--server-log", str(empty)]) == 1
    assert "is empty" in capsys.readouterr().err
    assert main(["kernel-cache", "check-log", "--server-log", str(missing)]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error: kernel-cache check-log: FileNotFoundError")
    assert captured.out == ""


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
OTHER_VARIANT_TENSOR = (
    "tt_dit_cache/model/transformer_ref/TP4_mesh4x8_bf16/blocks.0.weight.tensorbin"
)


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
    remote = dict(hub.files(WEIGHTS_REPO))
    remote_manifest = json.loads(remote.pop("tt_weights_manifest.json"))
    assert remote == {f"tt_dit_cache/{path}": data for path, data in TENSORS.items()}

    manifest = json.loads(manifest_out.read_text())
    assert manifest.pop("commit") == hub.commit(WEIGHTS_REPO)
    assert manifest == remote_manifest
    assert manifest["directories"] == [
        {"path": TEXT_ENCODER_DIR, "files": 1, "bytes": 7},
        {"path": TRANSFORMER_DIR, "files": 2, "bytes": 8},
    ]
    assert (manifest["files"], manifest["bytes"]) == (3, 15)
    assert manifest["meta"] == {"tt_metal_sha": "aaa", "server_sha": "bbb"}
    assert manifest["revision"] == "main"
    assert manifest["created_utc"].endswith("Z")
    assert hub.tags[WEIGHTS_REPO] == {"ttm-aaa-srv-bbb": hub.commit(WEIGHTS_REPO)}
    assert hub.called("upload_folder")[0]["revision"] == "main"


@pytest.mark.parametrize(
    "broken_file, removed_file, reason",
    [
        (None, f"{TEXT_ENCODER_DIR}/cache_dict.json", "no cache_dict.json"),
        (f"{TRANSFORMER_DIR}/x.tensorbin.tmp.4242.0", None, "1 temp file(s)"),
        ("stray.tensorbin", None, "1 file(s) outside a cache directory"),
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


def test_tt_weights_upload_refuses_a_symbolic_link(tt_dit_cache, tmp_path, capsys):
    host_file = tmp_path / "host_secret"
    host_file.write_text("JWT_SECRET=value")
    (tt_dit_cache / TRANSFORMER_DIR / "linked.tensorbin").symlink_to(host_file)
    hub = FakeHub()

    exit_code = main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub)

    assert exit_code == 1
    assert f"{TRANSFORMER_DIR}/linked.tensorbin" in capsys.readouterr().err
    assert hub.calls == []


def test_tt_weights_upload_refuses_an_existing_repo_that_is_not_private(
    tt_dit_cache, tmp_path, capsys
):
    hub = FakeHub({WEIGHTS_REPO: {}})
    hub.private[WEIGHTS_REPO] = False

    exit_code = main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub)

    assert exit_code == 1
    assert "exists and is not private" in capsys.readouterr().err
    assert hub.called("upload_folder") == []
    assert hub.files(WEIGHTS_REPO) == {}


def test_tt_weights_upload_replaces_its_directories_and_keeps_the_others(
    tt_dit_cache, tmp_path
):
    stale = f"tt_dit_cache/{TRANSFORMER_DIR}/renamed.tensorbin"
    hub = FakeHub(
        {
            WEIGHTS_REPO: {
                stale: b"old",
                OTHER_VARIANT_TENSOR: b"ref",
                "README.md": b"notes",
            }
        }
    )
    hub.tags[WEIGHTS_REPO] = {"ttm-old-srv-old": hub.commit(WEIGHTS_REPO)}

    exit_code = main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub)

    assert exit_code == 0
    assert hub.called("create_repo") == []
    remote = hub.files(WEIGHTS_REPO)
    assert stale not in remote
    assert remote[OTHER_VARIANT_TENSOR] == b"ref"
    assert remote["README.md"] == b"notes"
    assert set(hub.tags[WEIGHTS_REPO]) == {"ttm-old-srv-old", "ttm-aaa-srv-bbb"}


def test_tt_weights_upload_again_under_the_same_tag_reuses_what_is_published(
    tt_dit_cache, tmp_path, capsys
):
    hub = FakeHub()
    first, again = tmp_path / "first.json", tmp_path / "again.json"
    assert main(upload_argv(tt_dit_cache, first), hf_api=hub) == 0
    published = hub.commit(WEIGHTS_REPO)

    exit_code = main(upload_argv(tt_dit_cache, again), hf_api=hub)

    assert exit_code == 0
    assert json.loads(again.read_text()) == json.loads(first.read_text())
    assert len(hub.called("upload_folder")) == 1
    assert hub.commit(WEIGHTS_REPO) == published
    assert hub.tags[WEIGHTS_REPO] == {"ttm-aaa-srv-bbb": published}
    assert f"already holds this cache at commit {published}" in capsys.readouterr().out


def test_tt_weights_upload_refuses_to_move_an_existing_tag(
    tt_dit_cache, tmp_path, capsys
):
    hub = FakeHub()
    assert main(upload_argv(tt_dit_cache, tmp_path / "first.json"), hf_api=hub) == 0
    published = hub.commit(WEIGHTS_REPO)
    (tt_dit_cache / TRANSFORMER_DIR / "blocks.1.weight.tensorbin").write_bytes(b"pq")
    manifest_out = tmp_path / "manifest.json"

    exit_code = main(upload_argv(tt_dit_cache, manifest_out), hf_api=hub)

    assert exit_code == 1
    assert "tag ttm-aaa-srv-bbb already exists" in capsys.readouterr().err
    assert len(hub.called("upload_folder")) == 1
    assert hub.commit(WEIGHTS_REPO) == published
    assert not manifest_out.exists()


def test_tt_weights_interrupted_upload_cannot_be_downloaded_until_it_finishes(
    tt_dit_cache, tmp_path, capsys
):
    hub = FakeHub(batch=1)
    assert (
        main(upload_argv(tt_dit_cache, tmp_path / "old.json", "old"), hf_api=hub) == 0
    )
    old_release = tree(tt_dit_cache)
    for path, data in TENSORS.items():
        (tt_dit_cache / path).write_bytes(data.upper())
    hub.interrupt_after = 1
    upload_new = upload_argv(tt_dit_cache, tmp_path / "new.json", "new")
    from_branch, from_tag = tmp_path / "from_branch", tmp_path / "from_tag"

    assert main(upload_new, hf_api=hub) == 1

    assert "error: tt-weights upload: ConnectionError" in capsys.readouterr().err
    on_branch = [
        data
        for path, data in hub.files(WEIGHTS_REPO).items()
        if path.endswith(".tensorbin")
    ]
    assert {data.isupper() for data in on_branch} == {True, False}
    assert set(hub.tags[WEIGHTS_REPO]) == {"old"}
    assert main(download_argv(from_branch), hf_api=hub) == 1
    assert "an upload to it did not finish" in capsys.readouterr().err
    assert not from_branch.exists()
    assert main(download_argv(from_tag, revision="old"), hf_api=hub) == 0
    assert tree(from_tag) == old_release

    hub.interrupt_after = None
    assert main(upload_new, hf_api=hub) == 0
    assert main(download_argv(from_branch), hf_api=hub) == 0
    assert tree(from_branch) == tree(tt_dit_cache)


def test_tt_weights_upload_does_not_tag_a_commit_that_lacks_its_files(
    tt_dit_cache, tmp_path, capsys
):
    class OverlappingUploadHub(FakeHub):
        """Another upload to the revision removes a file before the manifest lands."""

        def upload_file(self, **kwargs):
            lost = f"tt_dit_cache/{TEXT_ENCODER_DIR}/embed.weight.tensorbin"
            del self.files(kwargs["repo_id"], kwargs["revision"])[lost]
            return super().upload_file(**kwargs)

    hub = OverlappingUploadHub()
    manifest_out = tmp_path / "manifest.json"

    exit_code = main(upload_argv(tt_dit_cache, manifest_out), hf_api=hub)

    assert exit_code == 1
    assert f"does not hold the local cache in {TEXT_ENCODER_DIR}" in (
        capsys.readouterr().err
    )
    assert hub.called("create_tag") == []
    assert not manifest_out.exists()


def test_tt_weights_download_verifies_then_marks_directories_complete(
    tt_dit_cache, tmp_path
):
    hub = FakeHub({WEIGHTS_REPO: {OTHER_VARIANT_TENSOR: b"ref"}})
    assert main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub) == 0
    dest = tmp_path / "deployment" / "tt_dit_cache"

    exit_code = main(download_argv(dest, revision="ttm-aaa-srv-bbb"), hf_api=hub)

    assert exit_code == 0
    assert tree(dest) == tree(tt_dit_cache)
    commit = hub.commit(WEIGHTS_REPO)
    assert hub.called("model_info")[-1]["revision"] == "ttm-aaa-srv-bbb"
    assert hub.called("hf_hub_download")[-1]["revision"] == commit
    download = hub.called("snapshot_download")[0]
    assert download["revision"] == commit
    assert download["allow_patterns"] == [
        f"tt_dit_cache/{TEXT_ENCODER_DIR}/*",
        f"tt_dit_cache/{TRANSFORMER_DIR}/*",
    ]


def test_tt_weights_download_that_fails_the_manifest_moves_nothing_and_unmarks(
    tt_dit_cache, tmp_path, capsys
):
    hub = FakeHub()
    assert main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub) == 0
    lost = f"tt_dit_cache/{TRANSFORMER_DIR}/blocks.0.bias.tensorbin"
    content = hub.files(WEIGHTS_REPO).pop(lost)
    dest = tmp_path / "deployment" / "tt_dit_cache"
    previous_release = {
        f"{TRANSFORMER_DIR}/blocks.0.weight.tensorbin": b"old",
        f"{TRANSFORMER_DIR}/blocks.0.bias.tensorbin": b"older",
    }
    write_files(dest, {**previous_release, f"{TRANSFORMER_DIR}/cache_dict.json": b""})

    exit_code = main(download_argv(dest), hf_api=hub)

    assert exit_code == 1
    assert f"{TRANSFORMER_DIR}: 1 files, 3 bytes" in capsys.readouterr().err
    assert not list(dest.rglob("cache_dict.json"))
    assert tree(dest / TRANSFORMER_DIR) == {
        Path(path).name: data for path, data in previous_release.items()
    }
    assert not (dest / TEXT_ENCODER_DIR).exists()

    hub.files(WEIGHTS_REPO)[lost] = content
    assert main(download_argv(dest), hf_api=hub) == 0
    assert tree(dest) == tree(tt_dit_cache)


def test_tt_weights_download_refuses_a_destination_with_files_of_another_release(
    tt_dit_cache, tmp_path, capsys
):
    hub = FakeHub()
    assert main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub) == 0
    dest = tmp_path / "deployment" / "tt_dit_cache"
    previous_release = {
        f"{TRANSFORMER_DIR}/renamed.tensorbin": b"old",
        f"{TRANSFORMER_DIR}/cache_dict.json": b"",
    }
    write_files(dest, previous_release)

    exit_code = main(download_argv(dest), hf_api=hub)

    assert exit_code == 1
    assert "not in the manifest, for example renamed.tensorbin" in (
        capsys.readouterr().err
    )
    assert not list(dest.rglob("cache_dict.json"))
    assert tree(dest / TRANSFORMER_DIR) == {"renamed.tensorbin": b"old"}


def test_tt_weights_download_that_stops_part_way_leaves_no_marker(
    tt_dit_cache, tmp_path, monkeypatch, capsys
):
    hub = FakeHub()
    assert main(upload_argv(tt_dit_cache, tmp_path / "manifest.json"), hf_api=hub) == 0
    dest = tmp_path / "deployment" / "tt_dit_cache"
    write_files(dest, {path: b"old" for path in tree(tt_dit_cache)})
    replace, moved = os.replace, []

    def replace_once(source, target):
        if moved:
            raise PermissionError(13, "Permission denied", str(target))
        moved.append(target)
        replace(source, target)

    monkeypatch.setattr(os, "replace", replace_once)

    exit_code = main(download_argv(dest), hf_api=hub)

    assert exit_code == 1
    assert capsys.readouterr().err.startswith(
        "error: tt-weights download: PermissionError"
    )
    assert len(moved) == 1
    assert not list(dest.rglob("cache_dict.json"))


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
        "commit": hub.commit(WEIGHTS_REPO),
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
