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
    model-info                                         what CI takes from the model's spec

What the tool has to know of a model is in its model spec (workflows/model_specs/),
under metadata.prebuilt_artifacts:

    kernel_build_env    env var names of the model that change the compiled kernels
    original_files      globs, relative to the checkpoint folder, of the files a
                        start with converted weights still reads
    original_shards     {index, tensor_prefix} entries: the index file and the
                        safetensors shards that, by that index, hold tensors
                        with the prefix are read too
    original_folders    folders that must exist in the destination, even when empty

kernel-cache and release-file work for any media-server model; without
kernel_build_env the manifest records BUILD_KEY_ENV_VARS only. tt-weights upload
needs original_files, also publishes the files of metadata.hf_extra_files and
assumes tt_dit's cache layout. scripts/tt_weights_download.py fetches what was
uploaded, for any model; build-image puts that file into the image. model-info
resolves a catalog model on a device as run.py does, fails when its spec has no
metadata.prebuilt_artifacts and else writes the variant name, the name of the
weights repo and the container folders of the converted weights and the
original checkpoint; it loads the catalog, which needs PyYAML as run.py does.

Run it from the repo root: python3 scripts/prebuilt_artifacts.py <command> --help
tt-weights upload needs huggingface_hub and takes the token from HF_TOKEN.
What a deployment sets is in tt-media-server/README.md, section "Image with
prebuilt kernels and weights".
"""

from __future__ import annotations

import argparse
import fnmatch
import functools
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
    VARIANT_SLUG,
    CliError,
    file_hash,
    gigabytes,
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

# Environment variables of tt-metal and the media server that change tt-metal's
# build key or kernel hashes: the ones a deployment is known to set. A model
# names its own in its spec (kernel_build_env). tt-metal hashes more options
# than these; a failing check-log is the sign that one is missing.
BUILD_KEY_ENV_VARS = (
    "TT_METAL_INSPECTOR",
    "TT_METAL_RISCV_DEBUG_INFO",
    "TT_MM_THROTTLE_PERF",
    "TT_METAL_WATCHER",
)
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
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
# Key of a model spec's metadata that holds what this tool has to know of the model.
SPEC_METADATA_KEY = "prebuilt_artifacts"
# The weights repo of a model is named after the repo of its original checkpoint.
WEIGHTS_REPO_SUFFIX = "-tt-weights"


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


def env_file_values(path: Path, recorded: list[str]) -> dict:
    """KEY=VALUE lines of a docker --env-file; values are literal, as docker reads them.

    A recorded variable must be given with its value.
    """
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.lstrip()
        if line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
        elif key.strip() in recorded:
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


def spec_metadata(path: Path) -> dict:
    """metadata of a runtime model spec JSON; empty when the spec has none."""
    metadata = read_model_spec(path).get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def env_name(value) -> bool:
    return isinstance(value, str) and bool(ENV_NAME.fullmatch(value))


def relative_path(value) -> bool:
    """True for a path or glob that stays inside the folder it is relative to."""
    if not isinstance(value, str) or value.startswith("/"):
        return False
    parts = PurePosixPath(value).parts
    return bool(parts) and ".." not in parts


def shard_rule(value) -> bool:
    """True for an original_shards entry: an index file and a tensor name prefix."""
    return (
        isinstance(value, dict)
        and value.keys() == {"index", "tensor_prefix"}
        and relative_path(value["index"])
        and isinstance(value["tensor_prefix"], str)
        and bool(value["tensor_prefix"])
    )


# The fields of metadata.prebuilt_artifacts and the check an item of each passes.
SPEC_METADATA_FIELDS = {
    "kernel_build_env": env_name,
    "original_files": relative_path,
    "original_shards": shard_rule,
    "original_folders": relative_path,
}


def spec_prebuilt_artifacts(path: Path) -> dict:
    """metadata.prebuilt_artifacts of a runtime model spec JSON; empty when the spec has none.

    The module docstring says what each field holds. Raises on any other field
    and on a field that is not a list of the items it takes.
    """
    described = spec_metadata(path).get(SPEC_METADATA_KEY, {})
    if not isinstance(described, dict):
        raise CliError(f"metadata.{SPEC_METADATA_KEY} of {path} is not a mapping")
    wrong = [
        name
        for name, items in described.items()
        if name not in SPEC_METADATA_FIELDS
        or not isinstance(items, list)
        or not all(map(SPEC_METADATA_FIELDS[name], items))
    ]
    if wrong:
        raise CliError(
            f"metadata.{SPEC_METADATA_KEY} of the model spec {path} has fields "
            f"this tool does not read as they are: {', '.join(wrong)}"
        )
    return described


def spec_extra_files(path: Path) -> list[str]:
    """Paths, relative to the checkpoint folder, of the files of metadata.hf_extra_files.

    Host setup fetches each from another repo into <local_subdir>/<filename>.
    """
    try:
        files = [
            (PurePosixPath(extra["local_subdir"]) / extra["filename"]).as_posix()
            for extra in spec_metadata(path).get("hf_extra_files", [])
        ]
    except (KeyError, TypeError) as error:
        raise CliError(
            f"metadata.hf_extra_files of the model spec {path} is not a list of "
            "entries with local_subdir and filename"
        ) from error
    outside = [file for file in files if not relative_path(file)]
    if outside:
        raise CliError(
            f"metadata.hf_extra_files of the model spec {path} names files "
            f"outside the checkpoint folder: {', '.join(outside)}"
        )
    return files


def read_kernel_manifest(path: Path, base_image: str) -> dict:
    """The kernel manifest at path; an error when it describes another base image.

    build_key_env maps the variables the capture run recorded to their values,
    None for an unset one.
    """
    manifest = manifest_fields(read_json(path), KERNEL_MANIFEST_FIELDS, path)
    if manifest["base_image"] != base_image:
        raise CliError(
            f"the kernel manifest {path} was made for base image "
            f"{manifest['base_image']}, not {base_image}"
        )
    build_key_env = manifest["build_key_env"]
    if not isinstance(build_key_env, dict) or not all(
        env_name(name) and (value is None or isinstance(value, str))
        for name, value in build_key_env.items()
    ):
        raise CliError(
            f"build_key_env of the kernel manifest {path} does not map variable "
            "names to a value or null"
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

    # Recorded: BUILD_KEY_ENV_VARS, then the variables the model spec names.
    recorded = list(BUILD_KEY_ENV_VARS)
    if spec_json:
        of_model = spec_prebuilt_artifacts(spec_json).get("kernel_build_env")
        if of_model is None:
            step(
                f"env: no metadata.{SPEC_METADATA_KEY}.kernel_build_env in "
                f"{spec_json}; no variable of the model is recorded"
            )
        recorded += [name for name in of_model or [] if name not in recorded]
    layers = [image_env(base_image)]
    not_given = []
    if env_file:
        layers.append(env_file_values(env_file, recorded))
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
    build_key_env = dict.fromkeys(recorded)
    for layer in layers:
        for name in recorded:
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
    """Dockerfile.kernel-cache plus the variables a kernel manifest records, in its order.

    A deployment that sets none of them then starts with the values the cache was
    compiled for, and hits it.
    """
    lines = [KERNEL_CACHE_DOCKERFILE.read_text(encoding="utf-8").rstrip("\n"), ""]
    for name, value in build_key_env.items():
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


def original_subset(
    original_dir: Path, described: dict, extra_files: list[str]
) -> tuple[list[dict], list[str]]:
    """Path, bytes and sha256 of the original files a model spec names, and the folders that must exist.

    The files are those the original_files globs match, each index of
    original_shards with the shards it gives for the prefix, and the extra
    files. Every glob and every named file must match a file.
    """
    wanted = list(described["original_files"])
    for rule in described.get("original_shards", []):
        index, prefix = PurePosixPath(rule["index"]), rule["tensor_prefix"]
        wanted.append(glob.escape(index.as_posix()))
        if not (original_dir / index).is_file():
            continue
        weight_map = read_json(original_dir / index).get("weight_map", {})
        shards = {
            (index.parent / str(shard)).as_posix()
            for name, shard in weight_map.items()
            if name.startswith(prefix)
        }
        if not shards:
            raise CliError(f"no {prefix}* weights in {original_dir / index}")
        wanted += [glob.escape(shard) for shard in sorted(shards)]
    wanted += [glob.escape(file) for file in extra_files]

    files, missing = set(), []
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
    return entries, list(described.get("original_folders", []))


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


def local_content(cache_dir: Path, original_dir: Path, manifest: dict) -> dict:
    """{repo path: (local path, bytes, sha256)} of every file a manifest publishes.

    Each converted file is read once, here; the manifest has the sha256 of the
    original files.
    """
    content = {}
    for entry in manifest["cache_directories"]:
        directory = cache_dir / entry["path"]
        for name, size in tensor_sizes(directory).items():
            content[f"{TT_WEIGHTS_FOLDER}/{entry['path']}/{name}"] = (
                directory / name,
                size,
                file_hash(directory / name),
            )
    for entry in manifest["original_files"]:
        content[f"{ORIGINAL_SUBSET_FOLDER}/{entry['path']}"] = (
            original_dir / entry["path"],
            entry["bytes"],
            entry["sha256"],
        )
    return content


def content_hash(content: dict) -> str:
    """One sha256 over the sha256 and repo path of every file of a local_content."""
    lines = (f"{sha256}  {path}\n" for path, (_, _, sha256) in sorted(content.items()))
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def differing(files: dict, content: dict, manifest: dict) -> tuple:
    """The manifest's cache directories and original files that a listing of the repo does not hold as they are locally.

    content is the local_content of the manifest. A file the Hub lists with a
    sha256 is compared by it; any other file is read for its git blob id.
    """

    def held(path: str) -> bool:
        local, size, sha256 = content[path]
        return same_content(local, size, files.get(path), sha256)

    directories = []
    for entry in manifest["cache_directories"]:
        folder = f"{TT_WEIGHTS_FOLDER}/{entry['path']}"
        local, remote = (
            {path for path in paths if path.rpartition("/")[0] == folder}
            for paths in (content, files)
        )
        if local != remote or not all(map(held, local)):
            directories.append(entry["path"])
    originals = [
        entry["path"]
        for entry in manifest["original_files"]
        if not held(f"{ORIGINAL_SUBSET_FOLDER}/{entry['path']}")
    ]
    return directories, originals


def manifest_lists(manifest: dict) -> list:
    """The directories, files and folders a weights manifest lists, each in an order that does not depend on the manifest."""
    key = functools.partial(json.dumps, sort_keys=True)
    fields = ("cache_directories", "original_files", "original_folders")
    return [sorted(manifest[field], key=key) for field in fields]


def published_version(
    api, repo, tags: dict, tag: str, manifest: dict, content: dict
) -> tuple | None:
    """(tag, commit, manifest) of a version of the variant that holds the local content, or None.

    Every tag whose commit has the manifest of the variant is a version of it,
    whatever its name. One holds the content when its manifest lists what the
    local one lists and the files at its commit have the local bytes. The given
    tag is tried first, then the tags named after the variant.
    """
    variant, lists, tried = manifest["variant"], manifest_lists(manifest), set()
    for name in sorted(
        tags, key=lambda name: (name != tag, not name.startswith(f"{variant}_"), name)
    ):
        commit = tags[name]
        if commit in tried:
            continue
        tried.add(commit)
        published = published_manifest(api, repo, commit, variant)
        if published is None or manifest_lists(published) != lists:
            continue
        if not any(differing(published_files(api, repo, commit), content, manifest)):
            return name, commit, published
    return None


def upload_plan(
    api, repo, revision, cache_dir, content, manifest, replace_shared
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

    directories, originals = differing(files, content, manifest)
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
    described = spec_prebuilt_artifacts(args.spec_json)
    if "original_files" not in described:
        raise CliError(
            f"no metadata.{SPEC_METADATA_KEY}.original_files in the model spec "
            f"{args.spec_json}: it lists the files of the original checkpoint a "
            "start with converted weights still reads, and without it nothing "
            "says what to publish"
        )
    extra_files = spec_extra_files(args.spec_json)
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
    originals, folders = original_subset(args.original_dir, described, extra_files)
    step(
        f"scan: {len(directories)} cache directories, "
        f"{sum(entry['files'] for entry in directories)} files, "
        f"{gigabytes(sum(entry['bytes'] for entry in directories))}; "
        f"{len(originals)} original files, "
        f"{gigabytes(sum(entry['bytes'] for entry in originals))}"
    )
    # The extra files come from other repos; the source revision is not theirs.
    checkpoint_files = [
        entry["path"] for entry in originals if entry["path"] not in extra_files
    ]
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

    Only content that is not published yet is uploaded: when a version of the
    variant holds the local bytes, whatever its tag, nothing is sent and the
    manifest written out is the one of that version. The repo holds other
    variants and they share directories. A path another variant's manifest lists
    is left as it is, or the upload is refused, or with --replace-shared that
    manifest is removed first; a file the Hub already holds is not sent again.
    The upload takes several commits, so this variant's manifest is removed
    first and written back last: a commit with the manifest is a finished
    upload. The tag is made only after the files at that commit have the content
    of the local ones. Tags are not moved: when the tag name holds other
    content, the new tag ends in the first 8 characters of the content hash.
    """
    api = hugging_face_api()
    repo, variant, branch = args.repo, args.variant, WEIGHTS_BRANCH
    manifest = describe_variant(args)
    directories, originals = manifest["cache_directories"], manifest["original_files"]
    content = local_content(args.cache_dir, args.original_dir, manifest)
    digest = content_hash(content)
    step(f"content: {len(content)} files, sha256 {digest}")
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
    held = published_version(api, repo, tags, tag, manifest, content)
    if held:
        tag, commit, manifest = held
        step(
            f"unchanged: {repo} holds this content as tag {tag} at commit "
            f"{commit}; nothing was uploaded"
        )
    else:
        if tag in tags:
            taken, tag = tag, f"{tag}_{digest[:8]}"
            step(
                f"tag: {taken} holds other content: the same inputs gave other "
                f"bytes; this version is tagged {tag}"
            )
        if tag in tags:
            raise CliError(
                f"tag {tag} already exists in {repo} with other content; tags "
                "are not moved"
            )
        api.create_branch(repo_id=repo, branch=branch, exist_ok=True)
        send_directories, stale, send_originals, retire = upload_plan(
            api, repo, branch, args.cache_dir, content, manifest, args.replace_shared
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
            published_files(api, repo, commit), content, manifest
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

    version = {"repo": repo, "tag": tag, "commit": commit, "uploaded": not held}
    write_json(args.manifest_out, {**manifest, **version})
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
            "image": {"reference": args.image, "digest": args.image_digest},
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
            "tests_outcome": args.tests_outcome,
            "warm_ready_seconds": args.warm_ready_seconds,
            "run_url": args.run_url,
        },
    )
    step(f"release_hash: {release_hash}")
    step(f"release file: {args.out}")


def write_model_info(args) -> None:
    """What CI takes from the spec of a catalog model, before it makes the artifacts.

    The spec is the one run.py resolves for the model on the device: the default
    implementation, from the dev catalog with --dev-mode. The catalog is loaded
    when workflows.model_spec is imported and needs PyYAML, so the import is
    here and the other commands run without it.
    """
    if args.dev_mode:
        os.environ["MODEL_SPECS_ENV"] = "dev"
    try:
        from workflows.model_spec import get_runtime_model_spec
        from workflows.setup_host import SetupConfig
        from workflows.utils import get_default_persistent_volume_root
    except ImportError as error:
        raise CliError(
            f"model-info loads the model catalog as run.py does, which needs "
            f"PyYAML: {error}"
        ) from error
    try:
        spec, _, _ = get_runtime_model_spec(model=args.model, device=args.device)
    except ValueError as error:
        raise CliError(f"{args.model} on {args.device}: {error}") from error
    model, device = spec.hf_model_repo, spec.device_type.name.lower()
    if not spec.metadata.get(SPEC_METADATA_KEY):
        raise CliError(
            f"the model spec of {model} on {device} has no "
            f"metadata.{SPEC_METADATA_KEY}: prebuilt artifacts are made only for "
            "a model whose spec has it"
        )
    variant = spec.model_name.lower()
    if not VARIANT_SLUG.fullmatch(variant):
        raise CliError(
            f"the name of {model} gives the variant '{variant}'; a variant is "
            "lowercase letters and digits joined by single hyphens"
        )
    if "TT_DIT_CACHE_DIR" not in spec.env_vars:
        raise CliError(
            f"no TT_DIT_CACHE_DIR in the env_vars of the model spec of {model} "
            f"on {device}"
        )
    # Host setup's paths for a run with --host-volume. The container side is the
    # same for any host folder.
    setup = SetupConfig(
        model_spec=spec,
        host_volume=str(get_default_persistent_volume_root(project_root)),
    )
    write_json(
        args.out,
        {
            "model": model,
            "device": device,
            "variant": variant,
            "weights_repo_name": PurePosixPath(spec.hf_weights_repo).name
            + WEIGHTS_REPO_SUFFIX,
            "converted_weights_dir": str(spec.env_vars["TT_DIT_CACHE_DIR"]),
            "original_dir": str(setup.container_model_weights_path),
        },
    )
    step(f"model info: {args.out}")


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
        help="runtime model spec JSON of the capture run, for its env_vars and "
        "the variables of the model its metadata names",
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
        f"still reads from the original checkpoint (the model spec names them "
        f"under metadata.{SPEC_METADATA_KEY}) and the manifest of the variant "
        f"to branch {WEIGHTS_BRANCH}, then tags the commit "
        f"<variant>_src-<source revision>_mesh<RxC>_ttm-<tt-metal commit>. Send "
        f"every kind of request the deployment serves before the upload: a "
        f"component no request loaded is not published. Run one upload to a repo "
        f"at a time. Only content that is not published yet is uploaded: when a "
        f"version of the variant holds the same bytes, whatever its tag, nothing "
        f"is sent and --manifest-out names that version. Tags are not moved; a "
        f"tag name that holds other content gets the first 8 characters of the "
        f"content hash appended. Needs huggingface_hub and the token in HF_TOKEN.",
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
        help="host directory the server read the original checkpoint from",
    )
    upload.add_argument(
        "--variant",
        type=variant_argument,
        required=True,
        help="lowercase name of the model variant, such as <model>-<task>; it "
        "names the manifest and starts the tag",
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
        help="runtime model spec JSON of that start; its metadata names the "
        "original files to publish",
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
        help="file to write the published manifest to, with repo, tag, commit "
        "and uploaded (false when the repo held the content)",
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
        "--tests-outcome",
        choices=("success", "failure"),
        help="outcome of the tests of the CI run; written as tests_outcome",
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

    info = command(
        commands,
        "model-info",
        write_model_info,
        "write what CI takes from the spec of a catalog model",
        f"Resolves the model on the device as run.py does, to its default "
        f"implementation, and writes a JSON file with model, device, variant "
        f"(the model name without its org, in lowercase), weights_repo_name "
        f"(the name of the spec's hf_weights_repo and {WEIGHTS_REPO_SUFFIX}), "
        f"converted_weights_dir (the spec's TT_DIT_CACHE_DIR) and original_dir "
        f"(where host setup puts the original checkpoint with --host-volume); "
        f"the last two are container paths. Exits 1 with one line when the "
        f"model is not in the catalog on that device or its spec has no "
        f"metadata.{SPEC_METADATA_KEY}. Needs PyYAML, as run.py does.",
    )
    info.add_argument(
        "--model", required=True, help="model, as run.py --model takes it"
    )
    info.add_argument(
        "--device", required=True, help="device type, as run.py --device takes it"
    )
    info.add_argument(
        "--dev-mode",
        action="store_true",
        help="read the dev catalog (workflows/model_specs/dev/), as run.py "
        "--dev-mode does",
    )
    info.add_argument(
        "--out", type=path_argument, required=True, help="JSON file to write"
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
