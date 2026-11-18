#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prebuilt start-up artifacts for a media-server model (MiniMax-H3 on Blackhole Galaxy).

A cold server start compiles the model's kernels and converts its weights. After
one such start this tool packages both, so later starts load them instead:

    kernel-cache manifest | build-image | check-log    compiled kernels, baked into the image
    tt-weights upload | download                       converted weights, in a Hugging Face repo
                                                       shared by the variants of the model
    release-hash                                       one JSON file that names both

Run it from the repo root: python3 scripts/prebuilt_artifacts.py <command> --help
The tt-weights commands need huggingface_hub and take the token from HF_TOKEN.
See tt-media-server/docs/minimax-h3-prebuilt-artifacts.md.

build-image copies this file alone into the image, where a deployment runs
tt-weights download with it. So nothing of the repo is imported at the top: the
commands that need workflows/kernel_cache.py import it when they run.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import fnmatch
import glob
import hashlib
import itertools
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

project_root = Path(__file__).resolve().parent.parent

# Environment variables that change tt-metal's build key or kernel hashes for this model.
BUILD_KEY_ENV_VARS = (
    "TT_METAL_INSPECTOR",
    "TT_METAL_RISCV_DEBUG_INFO",
    "TT_MM_THROTTLE_PERF",
    "MINIMAX_H3_TOPOLOGY",
    "MINIMAX_H3_VAE_OUTPUT",
    "TT_METAL_WATCHER",
)
KERNEL_CACHE_DOCKERFILE = project_root / "tt-media-server" / "Dockerfile.kernel-cache"
# Name of this file in the build context; the Dockerfile copies it into the image.
TOOL_IN_BUILD_CONTEXT = "prebuilt_artifacts.py"
# tt-metal logs "compiled <kernel>/<hash>/ in <N> ms" per compiled kernel; a cache hit logs nothing.
COMPILE_LINE = re.compile(r"compiled (\S+/[^/\s]+)/ in \d+ ms")
# tt-metal writes a cache file as <stem>.<random id>_<pid><ext> and renames it into place.
KERNEL_CACHE_TEMP_FILE = re.compile(r"\.\d+_\d+(\.|$)")

# One private repo holds every variant of a model family, side by side:
#   tt_dit_cache/<tree as under TT_DIT_CACHE_DIR>   converted weights, shared directories stored once
#   original_subset/<paths as in the checkpoint>    what a cached start still reads from the original
#   manifests/<variant>.json                        what one variant needs of the two
TT_WEIGHTS_FOLDER = "tt_dit_cache"
ORIGINAL_FOLDER = "original_subset"
MANIFEST_FOLDER = "manifests"
VARIANT_SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
# tt_dit loads a cache directory only when this (empty) file is in it.
CACHE_MARKER = "cache_dict.json"
TT_WEIGHTS_STAGING = ".tt_weights_download"
MESH_SHAPE = re.compile(r"mesh(\d+x\d+)_")
# Manifest fields that say when and by which server commit an upload was made,
# not what it holds. A tag is reused when everything else is equal.
MANIFEST_RUN_FIELDS = ("created_utc", "server_sha")
WEIGHTS_MANIFEST_FIELDS = (
    "variant",
    "cache_directories",
    "original_files",
    "original_folders",
)

# What a server with a complete converted cache still reads from the original
# checkpoint directory (MINIMAX_H3_MODEL_PATH). Correct the list here only.
# Globs relative to that directory; each must match a file. The audio VAE
# weights are listed for the audio encoder: only a request whose reference has a
# soundtrack loads it, so a capture run without one publishes no converted copy
# and the server converts it from these.
TEXT_ENCODER_INDEX = "text_encoder/model.safetensors.index.json"
ORIGINAL_SUBSET = (
    "model_index.json",
    "vae/config.json",
    "vae/*",
    "audio_vae/*.json",
    "audio_vae/*.safetensors",
    "text_encoder/config.json",
    TEXT_ENCODER_INDEX,
    "text_encoder/*.json",
    "tokenizer/**/*",
)
# config.json of the transformer folder(s) the directory holds:
TRANSFORMER_FOLDERS = ("transformer", "transformer_ref")
# the text encoder shard(s) that the index gives for weights with this prefix:
VISION_TOWER_PREFIX = "model.visual."
# and the LoRA file (MINIMAX_H3_LORA_PATH). These folders must exist, next to
# the transformer folder(s):
ORIGINAL_FOLDERS = ("text_encoder", "vae", "audio_vae")


class CliError(Exception):
    """A failure reported on stderr with exit code 1."""


def step(message: str) -> None:
    print(message, flush=True)


def kernel_cache_constants():
    """workflows/kernel_cache.py of the checkout this file is in.

    Imported by the commands that use it: the copy of this file in the image
    has no repo around it.
    """
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from workflows import kernel_cache

    return kernel_cache


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gigabytes(size: int) -> str:
    return f"{size / 1e9:.2f} GB"


def read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CliError(f"{path} is not valid JSON: {error}") from error


def manifest_fields(data, names: tuple, source) -> dict:
    """The named fields of a manifest; an error when it is not one of this layout."""
    missing = [name for name in names if not isinstance(data, dict) or name not in data]
    if missing:
        raise CliError(
            f"{source} is not a manifest this command reads: no {', '.join(missing)}"
        )
    return {name: data[name] for name in names}


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def docker(
    docker_run, args: list[str], capture: bool = False, env: dict | None = None
) -> str:
    command = ["docker", *args]
    result = docker_run(command, capture_output=capture, text=True, env=env)
    if result.returncode != 0:
        action = " ".join(itertools.takewhile(lambda arg: arg[0] != "-", command))
        detail = f": {result.stderr.strip()}" if capture and result.stderr else ""
        raise CliError(f"{action} failed{detail}")
    return result.stdout.strip() if capture else ""


def regular_files(root: Path) -> dict:
    """Bytes of every file under root, as {directory: {file name: bytes}}.

    The trees packaged here were written by containers. A symbolic link in one
    would pull a host file into a published image or repo, so anything that is
    not a regular file or a directory is refused.
    """
    tree, refused = {}, []
    for directory, folders, names in os.walk(root):
        folders.sort()
        directory = Path(directory)
        refused += [
            directory / name for name in folders if (directory / name).is_symlink()
        ]
        sizes = {}
        for name in sorted(names):
            status = (directory / name).lstat()
            if stat.S_ISREG(status.st_mode):
                sizes[name] = status.st_size
            else:
                refused.append(directory / name)
        tree[directory] = sizes
    if refused:
        shown = ", ".join(path.relative_to(root).as_posix() for path in refused[:5])
        raise CliError(
            f"symbolic links and special files are not packaged; {root} holds "
            f"{len(refused)}: {shown}"
        )
    return tree


def kernel_cache_files(cache_dir: Path) -> list:
    """(path, bytes) of the cache files an image needs: no compiler logs, no temp files."""
    return [
        (directory / name, size)
        for directory, sizes in regular_files(cache_dir).items()
        for name, size in sizes.items()
        if not (name.endswith(".log") or KERNEL_CACHE_TEMP_FILE.search(name))
    ]


def compiled_kernels(server_log: Path) -> list[str]:
    kernels = []
    with open(server_log, encoding="utf-8", errors="replace") as log:
        for line in log:
            match = COMPILE_LINE.search(line)
            if match:
                kernels.append(match.group(1))
    return kernels


def image_env(docker_run, image: str) -> dict:
    entries = json.loads(
        docker(
            docker_run,
            ["image", "inspect", "--format", "{{json .Config.Env}}", image],
            capture=True,
        )
    )
    return dict(entry.split("=", 1) for entry in entries or [] if "=" in entry)


def env_file_values(path: Path) -> dict:
    """KEY=VALUE lines of a docker --env-file; values are literal, as docker reads them."""
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.lstrip()
        if line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
        elif key.strip() in BUILD_KEY_ENV_VARS:
            raise CliError(
                f"{path} takes {key.strip()} from the environment of the docker "
                f"command, so its value in the capture run is not known; write it "
                f"as {key.strip()}=<value>"
            )
    return values


def read_model_spec(path: Path) -> dict:
    """A model spec JSON in the formats run.py reads: under runtime_model_spec or model_spec, or alone."""
    spec = read_json(path)
    return spec.get("runtime_model_spec") or spec.get("model_spec") or spec


def spec_env_values(path: Path) -> dict:
    """env_vars of a runtime model spec JSON, as run.py passes them with -e."""
    env_vars = read_model_spec(path).get("env_vars")
    if not isinstance(env_vars, dict):
        raise CliError(f"no env_vars in the model spec {path}")
    return {key: str(value) for key, value in env_vars.items() if value}


def kernel_cache_manifest(
    docker_run, cache_dir, server_log, base_image, env_file, spec_json, out
) -> None:
    kernel_roots = [path for path in cache_dir.glob("*/kernels") if path.is_dir()] or [
        path for path in cache_dir.glob("*/*/kernels") if path.is_dir()
    ]
    if not kernel_roots:
        raise CliError(f"no kernels folder under {cache_dir}: nothing was captured")
    build_keys = sorted(
        path.parent.relative_to(cache_dir).as_posix() for path in kernel_roots
    )
    kernel_hash_dirs = sum(
        1
        for key in build_keys
        for path in (cache_dir / key / "kernels").glob("*/*")
        if path.is_dir()
    )
    files = kernel_cache_files(cache_dir)
    size = sum(file_size for _, file_size in files)
    step(
        f"scan: {len(build_keys)} build key(s), {kernel_hash_dirs} kernel hash folders, "
        f"{len(files)} files, {gigabytes(size)} in {cache_dir}"
    )

    compiled = len(compiled_kernels(server_log))
    step(f"log: {compiled} kernels compiled in {server_log}")
    if kernel_hash_dirs and not compiled:
        raise CliError(
            f"{server_log} has no compile line although the cache holds "
            f"{kernel_hash_dirs} kernel hash folders: it is not the log of the "
            "capture run, or tt-metal did not log its compiles. check-log could "
            "not tell a cold start from a warm one."
        )

    layers = [image_env(docker_run, base_image)]
    not_given = []
    if env_file:
        layers.append(env_file_values(env_file))
    else:
        not_given.append("--env-file")
    if spec_json:
        layers.append(spec_env_values(spec_json))
    else:
        not_given.append("--spec-json")
    if not_given:
        step(
            f"env: no {' or '.join(not_given)} given; what the capture run set "
            "there is not in the manifest"
        )
    build_key_env = dict.fromkeys(BUILD_KEY_ENV_VARS)
    for layer in layers:
        for name in BUILD_KEY_ENV_VARS:
            if name in layer:
                build_key_env[name] = layer[name]
    step(
        "env: "
        + " ".join(
            f"{name}={'<unset>' if value is None else value}"
            for name, value in build_key_env.items()
        )
    )

    write_json(
        out,
        {
            "schema_version": 1,
            "created_utc": utc_now(),
            "base_image": base_image,
            "kernel_cache_path": kernel_cache_constants().KERNEL_CACHE_CONTAINER_DIR,
            "build_keys": build_keys,
            "kernel_hash_dirs": kernel_hash_dirs,
            "files": len(files),
            "bytes": size,
            "compiled_kernels_in_log": compiled,
            "build_key_env": build_key_env,
        },
    )
    step(f"manifest: {out}")


def render_kernel_cache_dockerfile(build_key_env: dict) -> str:
    """Dockerfile.kernel-cache plus the build-key variables of the capture run.

    A deployment that sets none of them then starts with the values the cache was
    compiled for, and hits it.
    """
    lines = [KERNEL_CACHE_DOCKERFILE.read_text(encoding="utf-8").rstrip("\n"), ""]
    for name in BUILD_KEY_ENV_VARS:
        value = build_key_env.get(name)
        if value is None:
            continue
        if any(ord(char) < 32 for char in value):
            raise CliError(f"{name} holds a control character and cannot be baked")
        escaped = re.sub(r'([\\"$])', r"\\\1", value)
        lines.append(f'ENV {name}="{escaped}"')
    return "\n".join(lines) + "\n"


def build_kernel_cache_image(
    docker_run, base_image, cache_dir, manifest_path, tag, push
) -> None:
    manifest = read_json(manifest_path)
    if manifest["base_image"] != base_image:
        raise CliError(
            f"the manifest was made for base image {manifest['base_image']}, "
            f"not {base_image}"
        )
    files = kernel_cache_files(cache_dir)
    size = sum(file_size for _, file_size in files)
    missing = [
        key
        for key in manifest["build_keys"]
        if not (cache_dir / key / "kernels").is_dir()
    ]
    if missing or (len(files), size) != (manifest["files"], manifest["bytes"]):
        raise CliError(
            f"{cache_dir} is not the kernel cache the manifest describes: it holds "
            f"{len(files)} files, {size} bytes"
            + (f" and has no {', '.join(missing)}" if missing else "")
            + f"; the manifest says {manifest['files']} files, {manifest['bytes']} bytes"
        )
    with tempfile.TemporaryDirectory(prefix="kernel-cache-image-") as context:
        context = Path(context)
        for source, _ in files:
            target = context / "kernel_cache" / source.relative_to(cache_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target, follow_symlinks=False)
        shutil.copyfile(manifest_path, context / "kernel_cache_manifest.json")
        shutil.copyfile(Path(__file__).resolve(), context / TOOL_IN_BUILD_CONTEXT)
        dockerfile = context / "Dockerfile"
        dockerfile.write_text(
            render_kernel_cache_dockerfile(manifest["build_key_env"]), encoding="utf-8"
        )
        step(f"stage: {len(files)} files, {gigabytes(size)} in {context}")

        step(f"build: {tag} from {base_image}")
        docker(
            docker_run,
            [
                "build",
                "--build-arg",
                f"BASE_IMAGE={base_image}",
                "--tag",
                tag,
                "--file",
                str(dockerfile),
                str(context),
            ],
            env={**os.environ, "DOCKER_BUILDKIT": "1"},
        )
    image_id = docker(
        docker_run, ["image", "inspect", "--format", "{{.Id}}", tag], capture=True
    )
    step(f"image-id: {image_id}")
    if not push:
        return

    step(f"push: {tag}")
    docker(docker_run, ["push", tag])
    repository = tag.rsplit(":", 1)[0] if ":" in tag.rsplit("/", 1)[-1] else tag
    repo_digests = json.loads(
        docker(
            docker_run,
            ["image", "inspect", "--format", "{{json .RepoDigests}}", tag],
            capture=True,
        )
    )
    digest = next(
        (entry for entry in repo_digests or [] if entry.split("@")[0] == repository),
        None,
    )
    if digest is None:
        raise CliError(f"docker reports no digest for {repository} after the push")
    step(f"digest: {digest}")


def check_log(server_log, max_compiled) -> int:
    if not server_log.stat().st_size:
        raise CliError(f"{server_log} is empty: it does not show a server start")
    kernels = compiled_kernels(server_log)
    if len(kernels) <= max_compiled:
        step(f"compiled kernels: {len(kernels)} (allowed: {max_compiled})")
        return 0
    step(
        f"compiled kernels: {len(kernels)}, more than the allowed {max_compiled}; "
        f"first {min(len(kernels), 20)}:"
    )
    for kernel in kernels[:20]:
        step(kernel)
    return 1


def hugging_face_api():
    """Hub client; huggingface_hub reads the token from HF_TOKEN itself."""
    if not os.environ.get("HF_TOKEN"):
        raise CliError("HF_TOKEN is not set")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise CliError("the tt-weights commands need huggingface_hub") from error
    return HfApi()


def file_hash(path: Path, digest=None) -> str:
    """sha256 of a file, or its hash in the digest given."""
    digest = digest or hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logged_cache_directories(server_log: Path, container_root: str) -> list[str]:
    """Directories under the container's TT_DIT_CACHE_DIR that a server log names.

    tt_dit logs the cache directory of every component it loads or writes.
    """
    named = re.compile(re.escape(container_root.rstrip("/")) + r"/([^\s'\"]+)")
    with open(server_log, encoding="utf-8", errors="replace") as log:
        return sorted(
            {path.rstrip("./") for line in log for path in named.findall(line)}
        )


def variant_cache_directories(
    cache_dir: Path, server_log: Path, container_root: str
) -> list[dict]:
    """Path, file count and bytes (without the marker) of the cache directories the log names.

    Raises unless every one of them is a finished conversion.
    """
    tree = {
        path.relative_to(cache_dir).as_posix(): sizes
        for path, sizes in regular_files(cache_dir).items()
        if sizes
    }
    named = logged_cache_directories(server_log, container_root)
    if not named:
        raise CliError(
            f"{server_log} names no directory under {container_root}: it is not "
            "the log of a server start that loaded or wrote the converted weights"
        )
    directories, problems = [], []
    for relative in named:
        sizes = tree.get(relative)
        if sizes is None:
            problems.append(f"{relative}: missing or empty")
            continue
        temp_files = fnmatch.filter(sizes, "*.tmp.*")
        if temp_files:
            problems.append(f"{relative}: {len(temp_files)} temp file(s)")
        if CACHE_MARKER not in sizes:
            problems.append(f"{relative}: no {CACHE_MARKER}")
        if any(other.startswith(f"{relative}/") for other in tree):
            problems.append(f"{relative}: holds a folder")
        tensors = [size for name, size in sizes.items() if name != CACHE_MARKER]
        directories.append(
            {"path": relative, "files": len(tensors), "bytes": sum(tensors)}
        )
    if problems:
        raise CliError(
            f"refusing to upload: the log names cache directories that are not "
            f"finished conversions in {cache_dir}:\n  " + "\n  ".join(problems)
        )
    left_out = sorted(set(tree) - set(named))
    if left_out:
        step(
            f"left out: {len(left_out)} directories the log does not name: "
            + ", ".join(left_out)
        )
    return directories


def lora_file(env_vars: dict) -> str | None:
    """The LoRA file of a model spec, relative to the checkpoint directory; None without one."""
    lora = PurePosixPath(env_vars.get("MINIMAX_H3_LORA_PATH", ""))
    model_path = PurePosixPath(env_vars.get("MINIMAX_H3_MODEL_PATH", ""))
    if not lora.name:
        return None
    if model_path not in lora.parents:
        raise CliError(
            "MINIMAX_H3_LORA_PATH is not inside MINIMAX_H3_MODEL_PATH in the model spec"
        )
    return lora.relative_to(model_path).as_posix()


def original_subset(
    original_dir: Path, lora: str | None
) -> tuple[list[dict], list[str]]:
    """Path, bytes and sha256 of the ORIGINAL_SUBSET files, and the folders that must exist."""
    transformers = [
        folder
        for folder in TRANSFORMER_FOLDERS
        if (original_dir / folder / "config.json").is_file()
    ]
    wanted = [*ORIGINAL_SUBSET, *(f"{folder}/config.json" for folder in transformers)]
    missing = (
        []
        if transformers
        else [" or ".join(f"{folder}/config.json" for folder in TRANSFORMER_FOLDERS)]
    )
    index = original_dir / TEXT_ENCODER_INDEX
    if index.is_file():
        shards = {
            shard
            for name, shard in read_json(index).get("weight_map", {}).items()
            if name.startswith(VISION_TOWER_PREFIX)
        }
        if not shards:
            raise CliError(f"no {VISION_TOWER_PREFIX}* weights in {index}")
        wanted += [f"text_encoder/{glob.escape(shard)}" for shard in sorted(shards)]
    if lora:
        wanted.append(glob.escape(lora))

    files = set()
    for pattern in wanted:
        matches = [path for path in original_dir.glob(pattern) if path.is_file()]
        files.update(matches)
        if not matches:
            missing.append(pattern)
    if missing:
        raise CliError(
            f"{original_dir} lacks files a start with converted weights reads: "
            + ", ".join(missing)
        )
    root = original_dir.resolve()
    links = sorted(
        path.relative_to(original_dir).as_posix()
        for path in files
        if path.resolve() != root / path.relative_to(original_dir)
    )
    if links:
        raise CliError(
            f"symbolic links are not published; {original_dir} reaches "
            f"{len(links)} file(s) through one: {', '.join(links[:5])}"
        )
    entries = [
        {
            "path": path.relative_to(original_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_hash(path),
        }
        for path in sorted(files)
    ]
    return entries, [*transformers, *ORIGINAL_FOLDERS]


def downloaded_revision(original_dir: Path, files: list[str]) -> str:
    """Commit the files were downloaded at, from the metadata `hf download --local-dir` keeps."""
    metadata = original_dir / ".cache" / "huggingface" / "download"
    revisions = set()
    for relative in files:
        record = metadata / f"{relative}.metadata"
        lines = record.read_text(encoding="utf-8").split() if record.is_file() else []
        revisions.add(lines[0] if lines else "unknown")
    return revisions.pop() if len(revisions) == 1 else "unknown"


def published_manifest(api, repo, commit, variant) -> dict | None:
    """manifests/<variant>.json at a commit of the repo, or None when it has none."""
    filename = f"{MANIFEST_FOLDER}/{variant}.json"
    if not api.file_exists(repo_id=repo, filename=filename, revision=commit):
        return None
    with tempfile.TemporaryDirectory(prefix="tt-weights-manifest-") as folder:
        manifest = read_json(
            api.hf_hub_download(
                repo_id=repo, filename=filename, revision=commit, local_dir=folder
            )
        )
    manifest_fields(
        manifest, WEIGHTS_MANIFEST_FIELDS, f"{filename} at {commit} of {repo}"
    )
    return manifest


def published_files(api, repo, commit) -> dict:
    """{path: file entry} of every file at a commit of the repo."""
    return {
        item.path: item
        for item in api.list_repo_tree(repo_id=repo, recursive=True, revision=commit)
        if getattr(item, "size", None) is not None
    }


def published_directory(files: dict, relative: str) -> dict:
    """{file name: file entry} of the files directly in one cache directory, in a listing of the repo."""
    folder = f"{TT_WEIGHTS_FOLDER}/{relative}"
    return {
        path[len(folder) + 1 :]: item
        for path, item in files.items()
        if path.rpartition("/")[0] == folder
    }


def same_content(path: Path, size: int, remote, sha256: str | None = None) -> bool:
    """True when a file entry of the Hub holds the bytes of the local file.

    The Hub reports a sha256 for a file in large-file storage and a git blob id
    for any other file.
    """
    if remote is None or remote.size != size:
        return False
    if remote.lfs:
        return remote.lfs.sha256 == (sha256 or file_hash(path))
    return remote.blob_id == file_hash(path, hashlib.sha1(b"blob %d\0" % size))


def tensor_sizes(directory: Path) -> dict:
    """Bytes per file of a cache directory, without its completion marker."""
    if not directory.is_dir():
        return {}
    return {
        item.name: item.stat().st_size
        for item in directory.iterdir()
        if item.is_file() and item.name != CACHE_MARKER
    }


def holds_directory(directory: Path, remote: dict) -> bool:
    """True when a local cache directory holds exactly the files the Hub lists for it."""
    local = tensor_sizes(directory)
    return (
        bool(remote)
        and remote.keys() == local.keys()
        and all(
            same_content(directory / name, size, remote[name])
            for name, size in local.items()
        )
    )


def is_listed_file(path: Path, entry: dict) -> bool:
    """True when a file has the bytes and sha256 of an original file of a manifest."""
    return (
        path.is_file()
        and path.stat().st_size == entry["bytes"]
        and file_hash(path) == entry["sha256"]
    )


def differing(files: dict, cache_dir, original_dir, manifest) -> tuple:
    """The manifest's cache directories and original files that a listing of the repo does not hold as they are locally."""
    directories = [
        entry["path"]
        for entry in manifest["cache_directories"]
        if not holds_directory(
            cache_dir / entry["path"], published_directory(files, entry["path"])
        )
    ]
    originals = [
        entry["path"]
        for entry in manifest["original_files"]
        if not same_content(
            original_dir / entry["path"],
            entry["bytes"],
            files.get(f"{ORIGINAL_FOLDER}/{entry['path']}"),
            entry["sha256"],
        )
    ]
    return directories, originals


def upload_plan(
    api, repo, revision, cache_dir, original_dir, manifest, replace_shared
) -> tuple:
    """What to send so the revision holds what the manifest lists.

    Returns the cache directories and original files that differ from the Hub,
    the Hub's files of those directories that the local ones lack, and the
    variants whose manifest leaves the revision first: this one and, with
    replace_shared, those that list a path that would change. Without
    replace_shared such a path is refused.
    """
    files = published_files(api, repo, revision)
    manifest_path = re.compile(rf"{MANIFEST_FOLDER}/([^/]+)\.json")
    variants = {
        match.group(1) for match in map(manifest_path.fullmatch, files) if match
    }
    listed = {}
    for variant in sorted(variants - {manifest["variant"]}):
        other = published_manifest(api, repo, revision, variant)
        paths = [
            f"{TT_WEIGHTS_FOLDER}/{entry['path']}"
            for entry in other["cache_directories"]
        ]
        paths += [
            f"{ORIGINAL_FOLDER}/{entry['path']}" for entry in other["original_files"]
        ]
        for path in paths:
            listed.setdefault(path, set()).add(variant)

    directories, originals = differing(files, cache_dir, original_dir, manifest)
    stale = [
        f"{path}/{name}"
        for path in directories
        for name in sorted(
            set(published_directory(files, path)) - set(tensor_sizes(cache_dir / path))
        )
    ]
    changed = [f"{TT_WEIGHTS_FOLDER}/{path}" for path in directories]
    changed += [f"{ORIGINAL_FOLDER}/{path}" for path in originals]
    refused = sorted(set(listed).intersection(changed))
    displaced = sorted(set().union(*(listed[path] for path in refused)))
    if refused and not replace_shared:
        raise CliError(
            f"{repo} holds other content at paths that the manifest of "
            f"{', '.join(displaced)} lists; they are not replaced and nothing was "
            "uploaded:\n  " + "\n  ".join(refused) + "\n--replace-shared replaces "
            f"them: it first removes those manifests from {revision}, until the "
            "variants are uploaded again. Their tags keep what they hold."
        )
    retire = sorted({manifest["variant"], *displaced} & variants)
    return directories, stale, originals, retire


def describe_variant(args) -> dict:
    """The manifest of one variant: what its server start used, and where that came from."""
    env_vars = spec_env_values(args.spec_json)
    if "TT_DIT_CACHE_DIR" not in env_vars:
        raise CliError(f"no TT_DIT_CACHE_DIR in the env_vars of {args.spec_json}")
    directories = variant_cache_directories(
        args.cache_dir, args.server_log, env_vars["TT_DIT_CACHE_DIR"]
    )
    meshes = sorted(
        {mesh for entry in directories for mesh in MESH_SHAPE.findall(entry["path"])}
    )
    if len(meshes) != 1:
        raise CliError(
            "the cache directory names must carry one mesh shape, found "
            + (", ".join(meshes) or "none")
        )
    lora = lora_file(env_vars)
    originals, folders = original_subset(args.original_dir, lora)
    step(
        f"scan: {len(directories)} cache directories, "
        f"{sum(entry['files'] for entry in directories)} files, "
        f"{gigabytes(sum(entry['bytes'] for entry in directories))}; "
        f"{len(originals)} original files, "
        f"{gigabytes(sum(entry['bytes'] for entry in originals))}"
    )
    checkpoint_files = [entry["path"] for entry in originals if entry["path"] != lora]
    return {
        "schema_version": 1,
        "variant": args.variant,
        "created_utc": utc_now(),
        "source": {
            "repo": read_model_spec(args.spec_json).get("hf_weights_repo") or "unknown",
            "revision": args.source_revision
            or downloaded_revision(args.original_dir, checkpoint_files),
        },
        "mesh_shape": meshes[0],
        "tt_metal_sha": args.tt_metal_sha,
        "server_sha": args.server_sha,
        "cache_directories": directories,
        "original_files": originals,
        "original_folders": folders,
    }


def upload_tt_weights(api, args) -> None:
    """Publish one variant: its cache directories, its original subset, its manifest and a tag.

    The repo holds other variants and they share directories. A path another
    variant's manifest lists is left as it is, or the upload is refused, or with
    --replace-shared that manifest is removed first; a file the Hub already
    holds is not sent again. The upload takes several commits, so this variant's
    manifest is removed first and written back last: a commit with the manifest
    is a finished upload. The tag is made only after the files at that commit
    have the content of the local ones. A tag that already holds this content is
    reused.
    """
    repo, revision, variant = args.repo, args.revision, args.variant
    manifest = describe_variant(args)
    directories, originals = manifest["cache_directories"], manifest["original_files"]
    tag = args.tag or (
        f"{variant}_src-{manifest['source']['revision'][:8]}"
        f"_mesh{manifest['mesh_shape']}_ttm-{args.tt_metal_sha[:8]}"
    )

    api.create_repo(repo_id=repo, private=True, exist_ok=True)
    if api.model_info(repo_id=repo).private is not True:
        raise CliError(
            f"{repo} exists and is not private; converted weights are only "
            "published to a private repo"
        )
    tags = {
        ref.name: ref.target_commit for ref in api.list_repo_refs(repo_id=repo).tags
    }
    if tag in tags:
        published = published_manifest(api, repo, tags[tag], variant) or {}
        if any(
            published.get(key) != value
            for key, value in manifest.items()
            if key not in MANIFEST_RUN_FIELDS
        ):
            raise CliError(
                f"tag {tag} already exists in {repo} with other content; "
                "tags are not moved"
            )
        step(f"tag: {tag} already holds this content at commit {tags[tag]}")
        commit, manifest = tags[tag], published
    else:
        api.create_branch(repo_id=repo, branch=revision, exist_ok=True)
        local = (args.cache_dir, args.original_dir, manifest)
        send_directories, stale, send_originals, retire = upload_plan(
            api, repo, revision, *local, args.replace_shared
        )
        for name in retire:
            path = f"{MANIFEST_FOLDER}/{name}.json"
            api.delete_file(
                path_in_repo=path,
                repo_id=repo,
                revision=revision,
                commit_message=f"Remove {path} until {name} is uploaded",
            )
        displaced = [name for name in retire if name != variant]
        if displaced:
            step(
                f"replace-shared: removed the manifest of {', '.join(displaced)} "
                f"from {revision}; upload them again"
            )
        if send_directories:
            api.upload_folder(
                repo_id=repo,
                folder_path=str(args.cache_dir),
                path_in_repo=TT_WEIGHTS_FOLDER,
                revision=revision,
                commit_message=f"Upload converted weights of {variant}",
                allow_patterns=[f"{glob.escape(path)}/*" for path in send_directories],
                ignore_patterns=[f"*/{CACHE_MARKER}"],
                delete_patterns=[glob.escape(path) for path in stale] or None,
            )
        if send_originals:
            api.upload_folder(
                repo_id=repo,
                folder_path=str(args.original_dir),
                path_in_repo=ORIGINAL_FOLDER,
                revision=revision,
                commit_message=f"Upload original subset of {variant}",
                allow_patterns=[glob.escape(path) for path in send_originals],
            )
        step(
            f"upload: {len(send_directories)} of {len(directories)} cache directories "
            f"and {len(send_originals)} of {len(originals)} original files to {repo} "
            f"at {revision}; the Hub already holds the rest"
        )
        commit = api.upload_file(
            path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
            path_in_repo=f"{MANIFEST_FOLDER}/{variant}.json",
            repo_id=repo,
            revision=revision,
        ).oid
        step(f"commit: {commit}")

        differ = sum(differing(published_files(api, repo, commit), *local), [])
        if differ:
            raise CliError(
                f"commit {commit} of {repo} does not hold the local files in "
                f"{', '.join(differ)}: something else changed {revision} during "
                "the upload. Nothing was tagged; run the upload again."
            )
        api.create_tag(repo_id=repo, tag=tag, revision=commit)
        step(f"tag: {tag}")

    write_json(
        args.manifest_out, {**manifest, "repo": repo, "tag": tag, "commit": commit}
    )
    step(f"manifest: {args.manifest_out}")


@contextlib.contextmanager
def exclusive(directory: Path):
    """Hold a download destination; a second download into it at the same time is refused."""
    directory.mkdir(parents=True, exist_ok=True)
    lock = os.open(directory, os.O_RDONLY)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CliError(
                f"another download is writing to {directory}; run this one after it"
            ) from error
        yield
    finally:
        os.close(lock)


def download_tt_weights(
    api, repo, revision, variant, dest_cache, dest_original
) -> None:
    """Fetch what manifests/<variant>.json of a revision lists and mark what was verified.

    What the destination already holds with the content of that revision is
    kept as it is, marker included: it is not fetched again, and a variant that
    shares it keeps working if this download fails. The markers of the other
    cache directories are removed first, so a download that stops part way
    leaves none of them looking complete. The files are checked in a staging
    folder before any of them is moved into place. A failed run keeps that
    folder, and the next run for the same commit continues from it.
    """
    commit = api.model_info(repo_id=repo, revision=revision).sha
    step(f"revision: {revision} is commit {commit}")

    manifest = published_manifest(api, repo, commit, variant)
    if manifest is None:
        raise CliError(
            f"{repo} has no {MANIFEST_FOLDER}/{variant}.json at {revision}: the "
            "variant is not published there, or an upload of it did not finish. "
            "Use the tag or commit of a release file."
        )
    listed, listed_files = manifest["cache_directories"], manifest["original_files"]
    folders = manifest["original_folders"]
    for path in [entry["path"] for entry in listed + listed_files] + folders:
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise CliError(f"unsafe path in the manifest: {path}")
    step(
        f"manifest: {len(listed)} cache directories, "
        f"{sum(entry['files'] for entry in listed)} files, "
        f"{gigabytes(sum(entry['bytes'] for entry in listed))}; "
        f"{len(listed_files)} original files"
    )

    files = published_files(api, repo, commit)
    directories = [
        entry
        for entry in listed
        if not holds_directory(
            dest_cache / entry["path"], published_directory(files, entry["path"])
        )
    ]
    originals = [
        entry
        for entry in listed_files
        if not is_listed_file(dest_original / entry["path"], entry)
    ]
    step(
        f"keep: {len(listed) - len(directories)} cache directories and "
        f"{len(listed_files) - len(originals)} original files that the destination "
        "already holds as they are at this commit"
    )

    markers = [dest_cache / entry["path"] / CACHE_MARKER for entry in directories]
    for marker in markers:
        marker.unlink(missing_ok=True)
    step(f"unmark: {len(markers)} directories in {dest_cache} until they are verified")

    staging = dest_cache / TT_WEIGHTS_STAGING / commit
    if staging.parent.exists() and not staging.is_dir():
        shutil.rmtree(staging.parent)
    patterns = [
        f"{TT_WEIGHTS_FOLDER}/{glob.escape(entry['path'])}/*" for entry in directories
    ] + [f"{ORIGINAL_FOLDER}/{glob.escape(entry['path'])}" for entry in originals]
    if patterns:
        api.snapshot_download(
            repo_id=repo,
            revision=commit,
            allow_patterns=patterns,
            local_dir=str(staging),
        )
    step(f"download: {variant} of {repo} into {staging}")

    staged = [staging / TT_WEIGHTS_FOLDER / entry["path"] for entry in directories]
    mismatches = []
    for entry, source, marker in zip(directories, staged, markers):
        sizes = tensor_sizes(source)
        other = sorted(set(tensor_sizes(marker.parent)) - set(sizes))
        if (len(sizes), sum(sizes.values())) != (entry["files"], entry["bytes"]):
            mismatches.append(
                f"{entry['path']}: {len(sizes)} files, {sum(sizes.values())} bytes; "
                f"the manifest says {entry['files']} files, {entry['bytes']} bytes"
            )
        elif other:
            mismatches.append(
                f"{entry['path']}: the destination holds {len(other)} file(s) "
                f"that are not in the manifest, for example {other[0]}"
            )
    for entry in originals:
        if not is_listed_file(staging / ORIGINAL_FOLDER / entry["path"], entry):
            mismatches.append(f"{entry['path']}: not the file the manifest lists")
    if mismatches:
        raise CliError(
            f"the download was not applied: nothing was moved into {dest_cache} or "
            f"{dest_original} and no directory is marked complete:\n  "
            + "\n  ".join(mismatches)
        )
    step(
        f"verify: {len(directories)} cache directories and {len(originals)} "
        "original files match the manifest"
    )

    for source, marker in zip(staged, markers):
        marker.parent.mkdir(parents=True, exist_ok=True)
        for name in tensor_sizes(source):
            os.replace(source / name, marker.parent / name)
    for entry in originals:
        target = dest_original / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staging / ORIGINAL_FOLDER / entry["path"], target)
    for folder in folders:
        (dest_original / folder).mkdir(parents=True, exist_ok=True)
    for entry in listed:
        (dest_cache / entry["path"] / CACHE_MARKER).touch()
    shutil.rmtree(staging.parent, ignore_errors=True)
    step(f"mark: {len(listed)} directories complete in {dest_cache}")


def write_release_hash(args) -> None:
    kernel = read_json(args.kernel_manifest)
    if kernel["base_image"] != args.base_image:
        raise CliError(
            f"the kernel manifest was made for base image {kernel['base_image']}, "
            f"not {args.base_image}"
        )
    constants = kernel_cache_constants()
    kernel_cache_env = {constants.KERNEL_CACHE_ENV: kernel["kernel_cache_path"]}
    kernel_cache_env.update(
        {
            name: value
            for name, value in kernel["build_key_env"].items()
            if value is not None
        }
    )

    tt_weights = None
    if args.weights_manifest:
        tt_weights = manifest_fields(
            read_json(args.weights_manifest),
            ("repo", "variant", "tag", "commit"),
            args.weights_manifest,
        )

    identity = {
        "model": args.model,
        "device": args.device,
        "tt_metal_sha": args.tt_metal_sha,
        "server_sha": args.server_sha,
        "image_digest": args.image_digest,
        "kernel_cache_env": kernel_cache_env,
        "tt_weights": tt_weights
        and {key: tt_weights[key] for key in ("repo", "variant", "commit")},
    }
    release_hash = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    write_json(
        args.out,
        {
            "schema_version": 1,
            "release_hash": release_hash,
            "created_utc": utc_now(),
            "model": args.model,
            "device": args.device,
            "tt_metal_sha": args.tt_metal_sha,
            "server_sha": args.server_sha,
            "image": {
                "reference": args.image,
                "digest": args.image_digest,
                "base_image": args.base_image,
            },
            "kernel_cache": {
                "path": kernel["kernel_cache_path"],
                "manifest_in_image": constants.KERNEL_CACHE_MANIFEST_CONTAINER_PATH,
                "env": kernel_cache_env,
                "build_keys": kernel["build_keys"],
                "kernel_hash_dirs": kernel["kernel_hash_dirs"],
                "files": kernel["files"],
                "bytes": kernel["bytes"],
            },
            "tt_weights": tt_weights,
            "warm_ready_seconds": args.warm_ready_seconds,
            "run_url": args.run_url,
        },
    )
    step(f"release-hash: {release_hash}")
    step(f"release: {args.out}")


def path_argument(value: str) -> Path:
    """A path option. An empty value would otherwise mean the current directory."""
    if not value.strip():
        raise argparse.ArgumentTypeError("the path is empty")
    return Path(value)


def variant_argument(value: str) -> str:
    """A variant slug. It names the manifest of the variant and starts its tag."""
    if not VARIANT_SLUG.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "a variant is a lowercase slug such as minimax-h3-fl2va-lightx2v"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package the compiled kernels and converted weights of a cold "
        "server start, so later starts load them."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    kernel_cache = commands.add_parser(
        "kernel-cache", help="compiled kernels, baked into the image"
    ).add_subparsers(dest="action", required=True)

    manifest = kernel_cache.add_parser(
        "manifest", help="describe a captured kernel cache"
    )
    manifest.add_argument("--cache-dir", type=path_argument, required=True)
    manifest.add_argument("--server-log", type=path_argument, required=True)
    manifest.add_argument("--base-image", required=True)
    manifest.add_argument("--env-file", type=path_argument)
    manifest.add_argument("--spec-json", type=path_argument)
    manifest.add_argument("--out", type=path_argument, required=True)

    build_image = kernel_cache.add_parser(
        "build-image", help="add a captured kernel cache to the image it came from"
    )
    build_image.add_argument("--base-image", required=True)
    build_image.add_argument("--cache-dir", type=path_argument, required=True)
    build_image.add_argument("--manifest", type=path_argument, required=True)
    build_image.add_argument("--tag", required=True)
    build_image.add_argument("--push", action="store_true")

    check = kernel_cache.add_parser(
        "check-log", help="fail when a server start compiled kernels"
    )
    check.add_argument("--server-log", type=path_argument, required=True)
    check.add_argument("--max-compiled", type=int, default=0)

    tt_weights = commands.add_parser(
        "tt-weights", help="converted weights, in a Hugging Face repo"
    ).add_subparsers(dest="action", required=True)

    upload = tt_weights.add_parser(
        "upload", help="publish the converted weights of one model variant"
    )
    upload.add_argument("--cache-dir", type=path_argument, required=True)
    upload.add_argument("--original-dir", type=path_argument, required=True)
    upload.add_argument("--variant", type=variant_argument, required=True)
    upload.add_argument("--server-log", type=path_argument, required=True)
    upload.add_argument("--spec-json", type=path_argument, required=True)
    upload.add_argument("--repo", required=True)
    upload.add_argument("--revision", default="main")
    upload.add_argument("--tt-metal-sha", required=True)
    upload.add_argument("--server-sha", required=True)
    upload.add_argument("--source-revision")
    upload.add_argument("--tag")
    upload.add_argument("--replace-shared", action="store_true")
    upload.add_argument("--manifest-out", type=path_argument, required=True)

    download = tt_weights.add_parser(
        "download", help="fetch and verify what one model variant needs"
    )
    download.add_argument("--repo", required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--variant", type=variant_argument, required=True)
    download.add_argument("--dest-cache", type=path_argument, required=True)
    download.add_argument("--dest-original", type=path_argument, required=True)

    release = commands.add_parser(
        "release-hash", help="write the file handed to the deployment team"
    )
    release.add_argument("--base-image", required=True)
    release.add_argument("--image", required=True)
    release.add_argument("--image-digest", required=True)
    release.add_argument("--kernel-manifest", type=path_argument, required=True)
    release.add_argument("--weights-manifest", type=path_argument)
    release.add_argument("--tt-metal-sha", required=True)
    release.add_argument("--server-sha", required=True)
    release.add_argument("--model", required=True)
    release.add_argument("--device", required=True)
    release.add_argument("--warm-ready-seconds", type=float)
    release.add_argument("--run-url")
    release.add_argument("--out", type=path_argument, required=True)
    return parser


def run(args, docker_run, hf_api) -> int:
    if args.command == "release-hash":
        write_release_hash(args)
    elif args.command == "kernel-cache":
        if args.action == "manifest":
            kernel_cache_manifest(
                docker_run,
                args.cache_dir,
                args.server_log,
                args.base_image,
                args.env_file,
                args.spec_json,
                args.out,
            )
        elif args.action == "build-image":
            build_kernel_cache_image(
                docker_run,
                args.base_image,
                args.cache_dir,
                args.manifest,
                args.tag,
                args.push,
            )
        else:
            return check_log(args.server_log, args.max_compiled)
    else:
        api = hf_api or hugging_face_api()
        if args.action == "upload":
            upload_tt_weights(api, args)
        else:
            with exclusive(args.dest_cache):
                download_tt_weights(
                    api,
                    args.repo,
                    args.revision,
                    args.variant,
                    args.dest_cache,
                    args.dest_original,
                )
    return 0


def main(argv=None, docker_run=subprocess.run, hf_api=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args, docker_run, hf_api)
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
    except OSError as error:
        command = " ".join(filter(None, (args.command, getattr(args, "action", None))))
        print(f"error: {command}: {type(error).__name__}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
