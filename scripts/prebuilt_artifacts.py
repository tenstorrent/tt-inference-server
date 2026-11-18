#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prebuilt start-up artifacts for a tt-media-server model.

A cold server start compiles the model's kernels and converts its weights. After
one such start this tool packages both, so later starts load them instead:

    kernel-cache manifest | build-image | check-log    compiled kernels, baked into the image
    tt-weights upload                                  converted weights, in a Hugging Face repo
                                                       shared by the variants of the model
    release-file                                       one JSON file that names both

kernel-cache and release-file work for any media-server model; the manifest
records BUILD_KEY_ENV_VARS, two of which are MiniMax-H3's. tt-weights upload
supports MiniMax-H3 only: it knows which files of that checkpoint a start with
converted weights still reads (the MINIMAX_H3_* block) and assumes tt_dit's
cache layout. scripts/tt_weights_download.py fetches what was uploaded, for any
model; build-image puts that file into the image.

Run it from the repo root: python3 scripts/prebuilt_artifacts.py <command> --help
tt-weights upload needs huggingface_hub and takes the token from HF_TOKEN.
What a deployment sets is in tt-media-server/README.md, section "MiniMax-H3
image with prebuilt kernels and weights".
"""

from __future__ import annotations

import argparse
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
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.tt_weights_download import (
    CACHE_MARKER,
    MANIFEST_FOLDER,
    ORIGINAL_SUBSET_FOLDER,
    SCHEMA_VERSION,
    TT_WEIGHTS_FOLDER,
    CliError,
    file_hash,
    gigabytes,
    holds_directory,
    hugging_face_api,
    manifest_fields,
    path_argument,
    published_directory,
    published_files,
    published_manifest,
    read_json,
    same_content,
    step,
    tensor_sizes,
    variant_argument,
)
from workflows.kernel_cache import (
    KERNEL_CACHE_CONTAINER_DIR,
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MANIFEST_CONTAINER_PATH,
)

# Environment variables that change tt-metal's build key or kernel hashes: the
# ones this deployment is known to set, four of tt-metal and the media server
# and two of MiniMax-H3. Another model adds its own here. tt-metal hashes more
# options than these; a failing check-log is the sign that one is missing.
BUILD_KEY_ENV_VARS = (
    "TT_METAL_INSPECTOR",
    "TT_METAL_RISCV_DEBUG_INFO",
    "TT_MM_THROTTLE_PERF",
    "MINIMAX_H3_TOPOLOGY",
    "MINIMAX_H3_VAE_OUTPUT",
    "TT_METAL_WATCHER",
)
KERNEL_CACHE_DOCKERFILE = project_root / "tt-media-server" / "Dockerfile.kernel-cache"
# build-image stages this file; it is the only tool file in the image.
DOWNLOAD_SCRIPT = project_root / "scripts" / "tt_weights_download.py"
# tt-metal logs "compiled <kernel>/<hash>/ in <N> ms" per compiled kernel when
# TT_METAL_LOG_KERNEL_COMPILE is on. tt_dit turns it on unless the variable is
# set, so a start with it set to 0 logs no such line. A cache hit logs nothing.
COMPILE_LINE = re.compile(r"compiled (\S+/[^/\s]+)/ in \d+ ms")
# tt-metal writes a cache file as <stem>.<random id>_<pid><ext> and renames it into place.
KERNEL_CACHE_TEMP_FILE = re.compile(r"\.\d+_\d+(\.|$)")
KERNEL_MANIFEST_FIELDS = (
    "base_image",
    "kernel_cache_path",
    "build_keys",
    "kernel_hash_dirs",
    "files",
    "bytes",
    "build_key_env",
)

# tt-weights upload writes to this branch of the weights repo and tags the result.
WEIGHTS_BRANCH = "main"
MESH_SHAPE = re.compile(r"mesh(\d+x\d+)_")
# Manifest fields that say when and by which server commit an upload was made,
# not what it holds. A tag is reused when everything else is equal.
MANIFEST_RUN_FIELDS = ("created_utc", "server_sha")

# MiniMax-H3 checkpoint layout, used by tt-weights upload only: what a server
# with a complete converted cache still reads from the original checkpoint
# directory (MINIMAX_H3_MODEL_PATH), besides the LoRA file (MINIMAX_H3_LORA_PATH).
# Correct the lists here only.

# Maps each text encoder weight to the shard that holds it.
MINIMAX_H3_TEXT_ENCODER_INDEX = "text_encoder/model.safetensors.index.json"
# Globs relative to the checkpoint directory; each must match a file. The audio
# VAE weights are listed for the audio encoder: only a request whose reference
# has a soundtrack loads it, so a capture run without one publishes no converted
# copy and the server converts it from these.
MINIMAX_H3_ORIGINAL_SUBSET = (
    "model_index.json",
    "vae/config.json",
    "vae/*",
    "audio_vae/*.json",
    "audio_vae/*.safetensors",
    "text_encoder/config.json",
    MINIMAX_H3_TEXT_ENCODER_INDEX,
    "text_encoder/*.json",
    "tokenizer/**/*",
)
# config.json is read from each of these folders that the checkpoint holds.
MINIMAX_H3_TRANSFORMER_FOLDERS = ("transformer", "transformer_ref")
# The text encoder shards that hold weights with this prefix are read.
MINIMAX_H3_VISION_TOWER_PREFIX = "model.visual."
# Folders that must exist, next to the transformer folder(s).
MINIMAX_H3_REQUIRED_FOLDERS = ("text_encoder", "vae", "audio_vae")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def docker(args: list[str], capture: bool = False, env: dict | None = None) -> str:
    command = ["docker", *args]
    result = subprocess.run(command, capture_output=capture, text=True, env=env)
    if result.returncode != 0:
        action = " ".join(itertools.takewhile(lambda arg: arg[0] != "-", command))
        detail = f": {result.stderr.strip()}" if capture and result.stderr else ""
        raise CliError(f"{action} failed{detail}")
    return result.stdout.strip() if capture else ""


def regular_files(root: Path) -> dict:
    """Bytes of every file under root, as {directory: {file name: bytes}}.

    The trees packaged here were written by containers. A symbolic link in one
    would pull a host file into a published image or repo, so anything that is
    not a regular file or a directory is refused. So is a folder that cannot be
    listed: its files would be left out.
    """

    def unreadable(error: OSError):
        raise CliError(f"cannot read {error.filename}: {error.strerror}")

    tree, refused = {}, []
    for directory, folders, names in os.walk(root, onerror=unreadable):
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


def image_env(image: str) -> dict:
    entries = json.loads(
        docker(
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
    return {key: str(value) for key, value in env_vars.items() if str(value)}


def read_kernel_manifest(path: Path, base_image: str) -> dict:
    """The kernel manifest at path; an error when it describes another base image."""
    manifest = manifest_fields(read_json(path), KERNEL_MANIFEST_FIELDS, path)
    if manifest["base_image"] != base_image:
        raise CliError(
            f"the kernel manifest {path} was made for base image "
            f"{manifest['base_image']}, not {base_image}"
        )
    return manifest


def kernel_cache_manifest(args) -> None:
    cache_dir, server_log, base_image = args.cache_dir, args.server_log, args.base_image
    env_file, spec_json, out = args.env_file, args.spec_json, args.out
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
            "capture run, or tt-metal did not log its compiles "
            "(TT_METAL_LOG_KERNEL_COMPILE=0). check-log could not tell a cold "
            "start from a warm one."
        )

    layers = [image_env(base_image)]
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
            "schema_version": SCHEMA_VERSION,
            "created_utc": utc_now(),
            "base_image": base_image,
            "kernel_cache_path": KERNEL_CACHE_CONTAINER_DIR,
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


def build_kernel_cache_image(args) -> None:
    base_image, cache_dir, tag = args.base_image, args.cache_dir, args.tag
    manifest = read_kernel_manifest(args.manifest, base_image)
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
        shutil.copyfile(args.manifest, context / "kernel_cache_manifest.json")
        shutil.copyfile(DOWNLOAD_SCRIPT, context / DOWNLOAD_SCRIPT.name)
        dockerfile = context / "Dockerfile"
        dockerfile.write_text(
            render_kernel_cache_dockerfile(manifest["build_key_env"]), encoding="utf-8"
        )
        step(f"stage: {len(files)} files, {gigabytes(size)} in {context}")

        step(f"build: {tag} from {base_image}")
        docker(
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
    image_id = docker(["image", "inspect", "--format", "{{.Id}}", tag], capture=True)
    step(f"image-id: {image_id}")


def check_log(args) -> int:
    server_log = args.server_log
    if not server_log.stat().st_size:
        raise CliError(f"{server_log} is empty: it does not show a server start")
    kernels = compiled_kernels(server_log)
    if not kernels:
        step("compiled kernels: 0")
        return 0
    step(f"compiled kernels: {len(kernels)}; first {min(len(kernels), 20)}:")
    for kernel in kernels[:20]:
        step(kernel)
    return 1


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

    Only those directories are scanned; the rest of cache_dir is not published.
    Raises unless every one of them is a finished conversion.
    """
    if not cache_dir.is_dir():
        raise CliError(f"{cache_dir} is not a directory")
    named = logged_cache_directories(server_log, container_root)
    if not named:
        raise CliError(
            f"{server_log} names no directory under {container_root}: it is not "
            "the log of a server start that loaded or wrote the converted weights"
        )
    root = cache_dir.resolve()
    directories, problems = [], []
    for relative in named:
        directory = cache_dir / relative
        if directory.resolve() != root / relative:
            problems.append(
                f"{relative}: outside {cache_dir} or behind a symbolic link"
            )
            continue
        tree = regular_files(directory) if directory.is_dir() else {}
        sizes = tree.pop(directory, None)
        if not sizes:
            problems.append(f"{relative}: missing or empty")
            continue
        temp_files = fnmatch.filter(sizes, "*.tmp.*")
        if temp_files:
            problems.append(f"{relative}: {len(temp_files)} temp file(s)")
        if CACHE_MARKER not in sizes:
            problems.append(f"{relative}: no {CACHE_MARKER}")
        if any(tree.values()):
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
    # The other finished conversions are listed by name; their files are not read.
    finished = {
        Path(folder).relative_to(cache_dir).as_posix()
        for folder, _, names in os.walk(cache_dir)
        if CACHE_MARKER in names
    }
    left_out = sorted(finished - set(named))
    if left_out:
        step(
            f"left out: {len(left_out)} directories the log does not name: "
            + ", ".join(left_out)
        )
    return directories


def minimax_h3_lora_file(env_vars: dict) -> str | None:
    """The LoRA file of a MiniMax-H3 model spec, relative to the checkpoint directory; None without one.

    The env_vars of another model are refused.
    """
    if "MINIMAX_H3_MODEL_PATH" not in env_vars:
        raise CliError("tt-weights upload supports MiniMax-H3 only")
    lora = PurePosixPath(env_vars.get("MINIMAX_H3_LORA_PATH", ""))
    model_path = PurePosixPath(env_vars["MINIMAX_H3_MODEL_PATH"])
    if not lora.name:
        return None
    if model_path not in lora.parents:
        raise CliError(
            "MINIMAX_H3_LORA_PATH is not inside MINIMAX_H3_MODEL_PATH in the model spec"
        )
    return lora.relative_to(model_path).as_posix()


def minimax_h3_original_subset(
    original_dir: Path, lora: str | None
) -> tuple[list[dict], list[str]]:
    """Path, bytes and sha256 of the MINIMAX_H3_ORIGINAL_SUBSET files, and the folders that must exist."""
    transformers = [
        folder
        for folder in MINIMAX_H3_TRANSFORMER_FOLDERS
        if (original_dir / folder / "config.json").is_file()
    ]
    wanted = [
        *MINIMAX_H3_ORIGINAL_SUBSET,
        *(f"{folder}/config.json" for folder in transformers),
    ]
    either_config = " or ".join(
        f"{folder}/config.json" for folder in MINIMAX_H3_TRANSFORMER_FOLDERS
    )
    missing = [] if transformers else [either_config]
    index = original_dir / MINIMAX_H3_TEXT_ENCODER_INDEX
    if index.is_file():
        shards = {
            shard
            for name, shard in read_json(index).get("weight_map", {}).items()
            if name.startswith(MINIMAX_H3_VISION_TOWER_PREFIX)
        }
        if not shards:
            raise CliError(f"no {MINIMAX_H3_VISION_TOWER_PREFIX}* weights in {index}")
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
    return entries, [*transformers, *MINIMAX_H3_REQUIRED_FOLDERS]


def downloaded_revision(original_dir: Path, files: list[str]) -> str:
    """Commit the files were downloaded at, from the records `hf download --local-dir` keeps.

    Raises unless every file has a record and all of them name one commit.
    """
    records = original_dir / ".cache" / "huggingface" / "download"
    revisions, unrecorded = set(), 0
    for relative in files:
        record = records / f"{relative}.metadata"
        lines = record.read_text(encoding="utf-8").split() if record.is_file() else []
        if lines:
            revisions.add(lines[0])
        else:
            unrecorded += 1
    if unrecorded or len(revisions) != 1:
        raise CliError(
            f"the download records under {records} do not give one source "
            f"revision: {unrecorded} of {len(files)} files have no record, "
            f"revisions found: {', '.join(sorted(revisions)) or 'none'}. Give the "
            "revision of the original checkpoint with --source-revision."
        )
    return revisions.pop()


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
            files.get(f"{ORIGINAL_SUBSET_FOLDER}/{entry['path']}"),
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
            f"{ORIGINAL_SUBSET_FOLDER}/{entry['path']}"
            for entry in other["original_files"]
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
    changed += [f"{ORIGINAL_SUBSET_FOLDER}/{path}" for path in originals]
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
    lora = minimax_h3_lora_file(env_vars)
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
    originals, folders = minimax_h3_original_subset(args.original_dir, lora)
    step(
        f"scan: {len(directories)} cache directories, "
        f"{sum(entry['files'] for entry in directories)} files, "
        f"{gigabytes(sum(entry['bytes'] for entry in directories))}; "
        f"{len(originals)} original files, "
        f"{gigabytes(sum(entry['bytes'] for entry in originals))}"
    )
    checkpoint_files = [entry["path"] for entry in originals if entry["path"] != lora]
    source_repo = read_model_spec(args.spec_json).get("hf_weights_repo")
    if not source_repo:
        step(
            f"source: no hf_weights_repo in {args.spec_json}; the manifest "
            "records the source repo as unknown"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "variant": args.variant,
        "created_utc": utc_now(),
        "source": {
            "repo": source_repo or "unknown",
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


def upload_tt_weights(args) -> None:
    """Publish one variant: its cache directories, its original subset, its manifest and a tag.

    The repo holds other variants and they share directories. A path another
    variant's manifest lists is left as it is, or the upload is refused, or with
    --replace-shared that manifest is removed first; a file the Hub already
    holds is not sent again. The upload takes several commits, so this variant's
    manifest is removed first and written back last: a commit with the manifest
    is a finished upload. The tag is made only after the files at that commit
    have the content of the local ones. A tag that already holds this content is
    reused; one that holds other content is an error, tags are not moved.
    """
    api = hugging_face_api()
    repo, variant, branch = args.repo, args.variant, WEIGHTS_BRANCH
    manifest = describe_variant(args)
    directories, originals = manifest["cache_directories"], manifest["original_files"]
    tag = (
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
        commit = tags[tag]
        published = published_manifest(api, repo, commit, variant) or {}
        changed = [
            key
            for key, value in manifest.items()
            if key not in MANIFEST_RUN_FIELDS and published.get(key) != value
        ]
        if not changed:
            changed_directories, changed_originals = differing(
                published_files(api, repo, commit),
                args.cache_dir,
                args.original_dir,
                manifest,
            )
            changed = changed_directories + changed_originals
        if changed:
            raise CliError(
                f"tag {tag} already exists in {repo} with other content "
                f"({', '.join(changed)}); tags are not moved"
            )
        step(f"tag: {tag} already holds this content at commit {commit}")
        manifest = published
    else:
        api.create_branch(repo_id=repo, branch=branch, exist_ok=True)
        send_directories, stale, send_originals, retire = upload_plan(
            api,
            repo,
            branch,
            args.cache_dir,
            args.original_dir,
            manifest,
            args.replace_shared,
        )
        for name in retire:
            path = f"{MANIFEST_FOLDER}/{name}.json"
            api.delete_file(
                path_in_repo=path,
                repo_id=repo,
                revision=branch,
                commit_message=f"Remove {path} until {name} is uploaded",
            )
        displaced = [name for name in retire if name != variant]
        if displaced:
            step(
                f"replace-shared: removed the manifest of {', '.join(displaced)} "
                f"from {branch}; upload them again"
            )
        if send_directories:
            api.upload_folder(
                repo_id=repo,
                folder_path=str(args.cache_dir),
                path_in_repo=TT_WEIGHTS_FOLDER,
                revision=branch,
                commit_message=f"Upload converted weights of {variant}",
                allow_patterns=[f"{glob.escape(path)}/*" for path in send_directories],
                ignore_patterns=[f"*/{CACHE_MARKER}"],
                delete_patterns=[glob.escape(path) for path in stale] or None,
            )
        if send_originals:
            api.upload_folder(
                repo_id=repo,
                folder_path=str(args.original_dir),
                path_in_repo=ORIGINAL_SUBSET_FOLDER,
                revision=branch,
                commit_message=f"Upload original subset of {variant}",
                allow_patterns=[glob.escape(path) for path in send_originals],
            )
        step(
            f"upload: {len(send_directories)} of {len(directories)} cache directories "
            f"and {len(send_originals)} of {len(originals)} original files to {repo} "
            f"at {branch}; the Hub already holds the rest"
        )
        commit = api.upload_file(
            path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
            path_in_repo=f"{MANIFEST_FOLDER}/{variant}.json",
            repo_id=repo,
            revision=branch,
        ).oid
        step(f"commit: {commit}")

        changed_directories, changed_originals = differing(
            published_files(api, repo, commit),
            args.cache_dir,
            args.original_dir,
            manifest,
        )
        if changed_directories or changed_originals:
            raise CliError(
                f"commit {commit} of {repo} does not hold the local files in "
                f"{', '.join(changed_directories + changed_originals)}: something "
                f"else changed {branch} during the upload. Nothing was tagged; run "
                "the upload again."
            )
        api.create_tag(repo_id=repo, tag=tag, revision=commit)
        step(f"tag: {tag}")

    write_json(
        args.manifest_out, {**manifest, "repo": repo, "tag": tag, "commit": commit}
    )
    step(f"manifest: {args.manifest_out}")


def write_release_file(args) -> None:
    kernel = read_kernel_manifest(args.kernel_manifest, args.base_image)
    kernel_cache_env = {KERNEL_CACHE_ENV: kernel["kernel_cache_path"]}
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
            "schema_version": SCHEMA_VERSION,
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
                "manifest_in_image": KERNEL_CACHE_MANIFEST_CONTAINER_PATH,
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
    step(f"release_hash: {release_hash}")
    step(f"release file: {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package the compiled kernels and converted weights of a cold "
        "server start, so later starts load them."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def command(group, name: str, handler, summary: str, details: str = ""):
        """A sub-command whose --help shows the summary and the details."""
        description = f"{summary[0].upper()}{summary[1:]}. {details}".strip()
        added = group.add_parser(name, help=summary, description=description)
        added.set_defaults(func=handler)
        return added

    kernel_cache = commands.add_parser(
        "kernel-cache", help="compiled kernels, baked into the image"
    ).add_subparsers(dest="action", required=True)

    manifest = command(
        kernel_cache,
        "manifest",
        kernel_cache_manifest,
        "describe a captured kernel cache",
        "Records its size and the variables that change the compiled kernels, as "
        "the capture run set them. Stops when the log has no compile line although "
        "kernels were captured, and on a symbolic link in the cache.",
    )
    manifest.add_argument(
        "--cache-dir",
        type=path_argument,
        required=True,
        help="host directory the capture run got as run.py --kernel-cache-dir",
    )
    manifest.add_argument(
        "--server-log",
        type=path_argument,
        required=True,
        help="server log of the capture run; its compile lines are counted",
    )
    manifest.add_argument(
        "--base-image", required=True, help="image the capture run started"
    )
    manifest.add_argument(
        "--env-file",
        type=path_argument,
        help="docker --env-file of the capture run, for the variables it set",
    )
    manifest.add_argument(
        "--spec-json",
        type=path_argument,
        help="runtime model spec JSON of the capture run, for its env_vars",
    )
    manifest.add_argument(
        "--out", type=path_argument, required=True, help="manifest file to write"
    )

    build_image = command(
        kernel_cache,
        "build-image",
        build_kernel_cache_image,
        "add a captured kernel cache to the image it came from",
        "Builds the image locally, without compiler logs and temporary files, and "
        "prints 'image-id: <id>'. It stages a copy of the cache in TMPDIR, which "
        "needs room for it.",
    )
    build_image.add_argument(
        "--base-image",
        required=True,
        help="image the cache was captured in; the manifest must name it",
    )
    build_image.add_argument(
        "--cache-dir",
        type=path_argument,
        required=True,
        help="captured kernel cache; it must be the one the manifest describes",
    )
    build_image.add_argument(
        "--manifest",
        type=path_argument,
        required=True,
        help="output of kernel-cache manifest; it is copied into the image",
    )
    build_image.add_argument("--tag", required=True, help="tag of the image to build")

    check = command(
        kernel_cache,
        "check-log",
        check_log,
        "fail when a server start compiled kernels",
        "Exits 0 when the log has no 'compiled <kernel>/<hash>/ in <N> ms' line, "
        "else 1 with the first 20 kernels. tt-metal writes that line only when "
        "TT_METAL_LOG_KERNEL_COMPILE is not 0 in the start. A missing or empty "
        "log fails.",
    )
    check.add_argument(
        "--server-log",
        type=path_argument,
        required=True,
        help="log of the server start to check",
    )

    tt_weights = commands.add_parser(
        "tt-weights", help="converted weights, in a Hugging Face repo"
    ).add_subparsers(dest="action", required=True)

    upload = command(
        tt_weights,
        "upload",
        upload_tt_weights,
        "publish the converted weights of one model variant",
        f"Sends the cache directories the server log names, the files a start "
        f"still reads from the original checkpoint and the manifest of the variant "
        f"to branch {WEIGHTS_BRANCH}, then tags the commit "
        f"<variant>_src-<source revision>_mesh<RxC>_ttm-<tt-metal commit>. Send "
        f"every kind of request the deployment serves before the upload: a "
        f"component no request loaded is not published. Run one upload to a repo "
        f"at a time. Tags are not moved; the same content under an existing tag "
        f"succeeds. Needs huggingface_hub and the token in HF_TOKEN.",
    )
    upload.add_argument(
        "--cache-dir",
        type=path_argument,
        required=True,
        help="host directory the server used as TT_DIT_CACHE_DIR",
    )
    upload.add_argument(
        "--original-dir",
        type=path_argument,
        required=True,
        help="host directory of the original checkpoint (MINIMAX_H3_MODEL_PATH)",
    )
    upload.add_argument(
        "--variant",
        type=variant_argument,
        required=True,
        help="lowercase name of the model variant, such as "
        "minimax-h3-fl2va-lightx2v; it names the manifest and starts the tag",
    )
    upload.add_argument(
        "--server-log",
        type=path_argument,
        required=True,
        help="log of the server start; the cache directories it names are published",
    )
    upload.add_argument(
        "--spec-json",
        type=path_argument,
        required=True,
        help="runtime model spec JSON of that start",
    )
    upload.add_argument(
        "--repo",
        required=True,
        help="private Hugging Face repo to publish to, as <org>/<name>; created "
        "private when missing",
    )
    upload.add_argument(
        "--tt-metal-sha",
        required=True,
        help="tt-metal commit the weights were validated with",
    )
    upload.add_argument(
        "--server-sha",
        required=True,
        help="tt-inference-server commit of the run; recorded in the manifest",
    )
    upload.add_argument(
        "--source-revision",
        help="revision of the original checkpoint; needed when the download "
        "records in --original-dir do not name one",
    )
    upload.add_argument(
        "--replace-shared",
        action="store_true",
        help="replace content at paths the manifest of another variant lists; "
        "that manifest leaves the branch until the variant is uploaded again",
    )
    upload.add_argument(
        "--manifest-out",
        type=path_argument,
        required=True,
        help="file to write the published manifest to, with repo, tag and commit",
    )

    release = command(
        commands,
        "release-file",
        write_release_file,
        "write the file handed to the deployment team",
        "It names the image and the weights that belong together. Its release_hash "
        "changes with the model, the device, the two commits, the image digest, "
        "the kernel cache variables and the weights repo, variant and commit, and "
        "with nothing else.",
    )
    release.add_argument(
        "--base-image",
        required=True,
        help="image the kernel cache was captured in; the kernel manifest must name it",
    )
    release.add_argument(
        "--image", required=True, help="reference of the image with the kernel cache"
    )
    release.add_argument(
        "--image-digest",
        required=True,
        help="registry digest of that image; deployments pull by it",
    )
    release.add_argument(
        "--kernel-manifest",
        type=path_argument,
        required=True,
        help="output of kernel-cache manifest",
    )
    release.add_argument(
        "--weights-manifest",
        type=path_argument,
        help="output of tt-weights upload; without it the release file names the "
        "image only",
    )
    release.add_argument(
        "--tt-metal-sha", required=True, help="tt-metal commit of the image"
    )
    release.add_argument(
        "--server-sha", required=True, help="tt-inference-server commit of the image"
    )
    release.add_argument("--model", required=True, help="model name")
    release.add_argument(
        "--device", required=True, help="device type the artifacts were made on"
    )
    release.add_argument(
        "--warm-ready-seconds",
        type=float,
        help="seconds the image took from start to ready in its validation start",
    )
    release.add_argument("--run-url", help="URL of the CI run that made the artifacts")
    release.add_argument(
        "--out", type=path_argument, required=True, help="release file to write"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args) or 0
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
    except OSError as error:
        command = " ".join(filter(None, (args.command, getattr(args, "action", None))))
        print(f"error: {command}: {type(error).__name__}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
