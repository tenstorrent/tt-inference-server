# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import fnmatch
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.prebuilt_artifacts import BUILD_KEY_ENV_VARS, KERNEL_CACHE_DOCKERFILE, main
from workflows.kernel_cache import KERNEL_CACHE_CONTAINER_DIR

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
    """Stands in for subprocess.run: answers image inspect and keeps the build context."""

    def __init__(self, image_env=()):
        self.image_env, self.commands, self.build_context = list(image_env), [], None

    def __call__(self, command, capture_output=False, text=False, env=None):
        self.commands.append(command)
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
        self.head = "0"

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
SET_VARIABLES = {
    "TT_METAL_INSPECTOR": "0",
    "TT_MM_THROTTLE_PERF": "5",
    "MINIMAX_H3_TOPOLOGY": "linear",
    "MINIMAX_H3_VAE_OUTPUT": "yuv420",
}
KERNEL_MANIFEST = {
    "base_image": BASE_IMAGE,
    "kernel_cache_path": KERNEL_CACHE_CONTAINER_DIR,
    "build_keys": ["tt-metal-cache4242"],
    "kernel_hash_dirs": 2,
    "files": 3,
    "bytes": 14,
    "compiled_kernels_in_log": 2,
    "build_key_env": {**dict.fromkeys(BUILD_KEY_ENV_VARS), **SET_VARIABLES},
}


@pytest.fixture
def kernel_cache(tmp_path):
    return write_files(tmp_path / "cache", {**IMAGE_FILES, **NOT_IMAGE_FILES})


def write_server_log(path, compiled):
    lines = [f"info | compiled k_{i}/{i}/ in 9 ms" for i in range(compiled)]
    path.write_text("\n".join(["info | opening mesh", *lines, "info | ready"]))
    return path


def test_kernel_cache_manifest_content_and_env_order(kernel_cache, tmp_path, capsys):
    docker = FakeDocker(["TT_MM_THROTTLE_PERF=5", "TT_METAL_INSPECTOR=1"])
    env = tmp_path / ".env"
    env.write_text("HF_TOKEN=hf_secret\nTT_METAL_INSPECTOR=0\nMINIMAX_H3_TOPOLOGY=ring")
    env_vars = {"MINIMAX_H3_TOPOLOGY": "linear", "MINIMAX_H3_VAE_OUTPUT": "yuv420"}
    spec_json = tmp_path / "runtime_model_spec.json"
    spec_json.write_text(json.dumps({"env_vars": {**env_vars, "TT_METAL_WATCHER": ""}}))
    out = tmp_path / "out" / "kernel_cache_manifest.json"
    argv = ["kernel-cache", "manifest", "--cache-dir", str(kernel_cache)]
    argv += ["--server-log", str(write_server_log(tmp_path / "server.log", 2))]
    argv += ["--base-image", BASE_IMAGE, "--env-file", str(env)]
    argv += ["--spec-json", str(spec_json), "--out", str(out)]

    assert main(argv, docker_run=docker) == 0

    manifest = json.loads(out.read_text())
    assert manifest.pop("created_utc").endswith("Z")
    assert manifest == {"schema_version": 1, **KERNEL_MANIFEST}
    assert "hf_secret" not in "".join(capsys.readouterr()) + out.read_text()


def build_image_argv(kernel_cache, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(KERNEL_MANIFEST))
    argv = ["kernel-cache", "build-image", "--base-image", BASE_IMAGE]
    argv += ["--cache-dir", str(kernel_cache), "--manifest", str(manifest)]
    return argv + ["--tag", KC_IMAGE]


def test_build_image_stages_cache_files_and_bakes_set_variables(kernel_cache, tmp_path):
    docker = FakeDocker()

    assert main(build_image_argv(kernel_cache, tmp_path), docker_run=docker) == 0

    staged = dict(docker.build_context)
    dockerfile = staged.pop("Dockerfile").decode()
    assert json.loads(staged.pop("kernel_cache_manifest.json")) == KERNEL_MANIFEST
    assert staged == {
        f"kernel_cache/{path}": data for path, data in IMAGE_FILES.items()
    }
    assert dockerfile.startswith(KERNEL_CACHE_DOCKERFILE.read_text())
    baked = [f'ENV {name}="{value}"' for name, value in SET_VARIABLES.items()]
    env_lines = [line for line in dockerfile.splitlines() if line.startswith("ENV ")]
    assert env_lines == [f"ENV TT_METAL_CACHE={KERNEL_CACHE_CONTAINER_DIR}", *baked]


def test_build_image_refuses_a_tree_that_differs(kernel_cache, tmp_path, capsys):
    (kernel_cache / "tt-metal-cache4242/kernels/reader/111/added.o").write_bytes(b"x")
    docker = FakeDocker()

    assert main(build_image_argv(kernel_cache, tmp_path), docker_run=docker) == 1

    error = capsys.readouterr().err
    assert "holds 4 files, 15 bytes; the manifest says 3 files, 14 bytes" in error
    assert docker.commands == []


@pytest.mark.parametrize("compiled, exit_code", [(0, 0), (25, 1)])
def test_check_log_fails_on_compiled_kernels(tmp_path, capsys, compiled, exit_code):
    argv = ["kernel-cache", "check-log", "--server-log"]
    argv.append(str(write_server_log(tmp_path / "server.log", compiled)))

    assert main(argv) == exit_code

    output = capsys.readouterr().out.splitlines()
    assert output[0].startswith(f"compiled kernels: {compiled}")
    assert output[1:] == [f"k_{i}/{i}" for i in range(min(compiled, 20))]


FL2VA, REF2VA = "minimax-h3-fl2va-lightx2v", "minimax-h3-ref2va-lightx2v"
CONTAINER_CACHE = "/home/container_app_user/cache_root/tt_dit_cache"
MODEL_PATH = "/home/container_app_user/cache_root/weights/Example-Model"
SOURCE_COMMIT = "c0ffee12" + "0" * 32
SHARED = "model/text_encoder/mesh4x8_bf16"
OWN = {
    FL2VA: "model/transformer_lora-a/TP4_mesh4x8_bf16",
    REF2VA: "model/transformer_ref_lora-b/TP4_mesh4x8_bf16",
}
NOT_LOADED = "model/vae_previous/mesh1x8_fp32"
CACHE_FILES = {
    f"{OWN[FL2VA]}/blocks.0.weight.tensorbin": b"abc",
    f"{OWN[FL2VA]}/blocks.0.bias.tensorbin": b"defgh",
    f"{OWN[REF2VA]}/blocks.0.weight.tensorbin": b"ref",
    f"{SHARED}/embed.weight.tensorbin": b"ijklmno",
    f"{NOT_LOADED}/conv.weight.tensorbin": b"stale",
}
SHARDS = {
    "model.language_model.layers.0.weight": "model-00001-of-00002.safetensors",
    "model.visual.blocks.0.weight": "model-00002-of-00002.safetensors",
}
INDEX = json.dumps({"weight_map": SHARDS}).encode()
LOG = f"../{FL2VA}/server.log"
NOT_READ_BY_A_CACHED_START = {
    "transformer/diffusion_pytorch_model.safetensors": b"big",
    "text_encoder/model-00001-of-00002.safetensors": b"big",
    "lightx2v/another-adapter.safetensors": b"big",
}


def original_subset(variant):
    transformer = "transformer_ref" if variant == REF2VA else "transformer"
    return {
        "model_index.json": b"{}",
        "tokenizer/tokenizer.json": b"{}",
        f"{transformer}/config.json": b"{}",
        "text_encoder/config.json": b"{}",
        "text_encoder/preprocessor_config.json": b"{}",
        "text_encoder/model.safetensors.index.json": INDEX,
        "text_encoder/model-00002-of-00002.safetensors": b"vision",
        "vae/config.json": b"{}",
        "vae/diffusion_pytorch_model.safetensors": b"vae",
        "audio_vae/config.json": b"{}",
        f"lightx2v/{variant}.safetensors": variant.encode(),
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


def upload_argv(cache_dir, variant=FL2VA, tt_metal_sha="a" * 40, server_sha="b" * 40):
    """Upload after a start of the variant: shared directory loaded, its own one written."""
    work = cache_dir.parent / variant
    files = {**original_subset(variant), **NOT_READ_BY_A_CACHED_START}
    for path in original_subset(variant):
        if not path.startswith("lightx2v/"):
            metadata = f".cache/huggingface/download/{path}.metadata"
            files[metadata] = f"{SOURCE_COMMIT}\netag\n0.0\n".encode()
    write_files(work / "weights", files)
    (work / "server.log").write_text(
        f"INFO | loading 1 cached weight tensors from '{CONTAINER_CACHE}/{SHARED}'...\n"
        "INFO | Cache does not exist. Loading PyTorch state dict.\n"
        f"INFO | Writing cache to '{CONTAINER_CACHE}/{OWN[variant]}'.\n"
    )
    env_vars = {
        "TT_DIT_CACHE_DIR": CONTAINER_CACHE,
        "MINIMAX_H3_MODEL_PATH": MODEL_PATH,
        "MINIMAX_H3_LORA_PATH": f"{MODEL_PATH}/lightx2v/{variant}.safetensors",
    }
    spec = {"hf_weights_repo": "example-org/original-model", "env_vars": env_vars}
    (work / "spec.json").write_text(json.dumps({"runtime_model_spec": spec}))
    argv = ["tt-weights", "upload", "--cache-dir", str(cache_dir), "--variant", variant]
    argv += ["--original-dir", str(work / "weights"), "--repo", WEIGHTS_REPO]
    argv += ["--server-log", str(work / "server.log")]
    argv += ["--spec-json", str(work / "spec.json"), "--tt-metal-sha", tt_metal_sha]
    return argv + ["--server-sha", server_sha, "--manifest-out", str(work / "out.json")]


@pytest.fixture
def hub(cache_dir, capsys):
    hub = FakeHub()
    assert main(upload_argv(cache_dir), hf_api=hub) == 0
    return hub


@pytest.mark.parametrize(
    "option, downloaded_with_hf, source",
    [([], True, SOURCE_COMMIT), ([], False, "unknown"), (["f" * 9], True, "f" * 9)],
)
def test_upload_publishes_what_the_log_names_and_tags_it_by_its_source(
    cache_dir, capsys, option, downloaded_with_hf, source
):
    hub = FakeHub()
    argv = upload_argv(cache_dir) + ["--source-revision"] * len(option) + option
    weights = cache_dir.parent / FL2VA / "weights"
    if not downloaded_with_hf:
        (weights / ".cache").rename(weights.parent / "moved")

    assert main(argv, hf_api=hub) == 0

    assert hub.private is True
    remote = dict(hub.files())
    manifest = json.loads(remote.pop(f"manifests/{FL2VA}.json"))
    assert remote == published(FL2VA)
    left_out = f"the log does not name: {OWN[REF2VA]}, {NOT_LOADED}\n"
    assert left_out in capsys.readouterr().out
    tag = f"{FL2VA}_src-{source[:8]}_mesh4x8_ttm-aaaaaaaa"
    assert hub.tags == {tag: hub.head}
    out = json.loads((weights.parent / "out.json").read_text())
    assert out == {**manifest, "repo": WEIGHTS_REPO, "tag": tag, "commit": hub.head}
    assert manifest.pop("created_utc").endswith("Z")
    sha256 = {path: hashlib.sha256(data).hexdigest() for path, data in remote.items()}
    assert manifest == {
        "schema_version": 1,
        "variant": FL2VA,
        "source": {"repo": "example-org/original-model", "revision": source},
        "mesh_shape": "4x8",
        "tt_metal_sha": "a" * 40,
        "server_sha": "b" * 40,
        "cache_directories": [
            {"path": SHARED, "files": 1, "bytes": 7},
            {"path": OWN[FL2VA], "files": 2, "bytes": 8},
        ],
        "original_files": [
            {"path": p, "bytes": len(d), "sha256": sha256[f"original_subset/{p}"]}
            for p, d in sorted(original_subset(FL2VA).items())
        ],
        "original_folders": ["transformer", "text_encoder", "vae", "audio_vae"],
    }


def test_upload_stores_shared_paths_once_and_replaces_only_its_own(hub, cache_dir):
    hub.sent.clear()
    assert main(upload_argv(cache_dir, REF2VA), hf_api=hub) == 0
    assert set(hub.sent) == set(published(REF2VA)) - set(published(FL2VA))
    own = cache_dir / OWN[REF2VA]
    (own / "blocks.0.weight.tensorbin").rename(own / "blocks.0.scale.tensorbin")
    assert main(upload_argv(cache_dir, REF2VA, tt_metal_sha="c" * 40), hf_api=hub) == 0
    remote = {path for path in hub.files() if not path.startswith("manifests/")}
    renamed = {path.replace("0.weight", "0.scale") for path in published(REF2VA)}
    assert remote == renamed | set(published(FL2VA))


@pytest.mark.parametrize(
    "exit_code, tt_metal, path, content, message",
    [
        (0, "a", f"{NOT_LOADED}/x.tensorbin", b"", "already holds this content at"),
        (1, "a", f"{OWN[FL2VA]}/x.tensorbin", b"", "already exists in"),
        (1, "c", f"{SHARED}/x.tensorbin", b"", "another variant's manifest lists"),
        (1, "c", f"{SHARED}/x.tmp.42.0", b"", "1 temp file(s)"),
        (1, "c", f"{SHARED}/cache_dict.json", None, "no cache_dict.json"),
        (1, "c", f"{SHARED}/link", Path("/"), "symbolic links"),
        (1, "c", LOG, b"ready", "names no directory under"),
        (1, "c", LOG, f"'{CONTAINER_CACHE}/gone'".encode(), "gone: missing"),
        (1, "c", f"../{FL2VA}/weights/vae/config.json", None, "reads: vae/config.json"),
    ],
)
def test_upload_that_must_not_change_the_hub(
    hub, cache_dir, capsys, exit_code, tt_metal, path, content, message
):
    assert main(upload_argv(cache_dir, REF2VA), hf_api=hub) == 0
    argv = upload_argv(cache_dir, tt_metal_sha=tt_metal * 40, server_sha="d" * 40)
    if isinstance(content, Path):
        (cache_dir / path).symlink_to(content)
    elif content is None:
        (cache_dir / path).unlink()
    else:
        (cache_dir / path).write_bytes(content)
    before = (hub.head, dict(hub.tags))

    assert main(argv, hf_api=hub) == exit_code

    assert message in "".join(capsys.readouterr())
    assert (hub.head, hub.tags) == before


def download_argv(dest):
    argv = ["tt-weights", "download", "--repo", WEIGHTS_REPO, "--revision", "main"]
    argv += ["--variant", FL2VA, "--dest-cache", str(dest / "tt_dit_cache")]
    return argv + ["--dest-original", str(dest / "weights")]


def test_download_marks_one_variant_after_verifying(hub, cache_dir, tmp_path):
    assert main(upload_argv(cache_dir, REF2VA), hf_api=hub) == 0
    dest = tmp_path / "deployment"

    assert main(download_argv(dest), hf_api=hub) == 0

    expected = published(FL2VA, original="weights")
    for directory in (OWN[FL2VA], SHARED):
        expected[f"tt_dit_cache/{directory}/cache_dict.json"] = b""
    assert tree(dest) == expected


@pytest.mark.parametrize(
    "path, content",
    [
        (f"tt_dit_cache/{OWN[FL2VA]}/blocks.0.bias.tensorbin", None),
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

    assert main(download_argv(dest), hf_api=hub) == 1

    assert "the download was not applied" in capsys.readouterr().err
    assert not list(dest.rglob("cache_dict.json"))
    assert tree(dest / "tt_dit_cache" / SHARED) == {"embed.weight.tensorbin": b"old"}
    assert not (dest / "weights").exists()


def test_release_hash_names_image_kernel_cache_and_weights(tmp_path):
    kernel_manifest = tmp_path / "kernel_cache_manifest.json"
    kernel_manifest.write_text(json.dumps(KERNEL_MANIFEST))
    weights = {"repo": WEIGHTS_REPO, "variant": FL2VA, "tag": "t", "commit": "1" * 40}
    weights_manifest = tmp_path / "tt_weights_manifest.json"
    weights_manifest.write_text(json.dumps({**weights, "cache_directories": []}))
    out = tmp_path / "release.json"
    argv = ["release-hash", "--base-image", BASE_IMAGE, "--image", KC_IMAGE]
    argv += ["--image-digest", "sha256:2222", "--kernel-manifest", str(kernel_manifest)]
    argv += ["--weights-manifest", str(weights_manifest), "--tt-metal-sha", "aaa"]
    argv += ["--server-sha", "bbb", "--model", "example-model", "--device", "galaxy"]

    assert main(argv + ["--out", str(out)]) == 0

    release = json.loads(out.read_text())
    assert len(release["release_hash"]) == 64
    assert release["tt_weights"] == weights
    image = {"reference": KC_IMAGE, "digest": "sha256:2222", "base_image": BASE_IMAGE}
    assert release["image"] == image
    kernel_cache_env = {"TT_METAL_CACHE": KERNEL_CACHE_CONTAINER_DIR, **SET_VARIABLES}
    assert release["kernel_cache"]["env"] == kernel_cache_env
