#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prebuilt start-up artifacts for a media-server model (MiniMax-H3 on Blackhole Galaxy).

A cold server start compiles the model's kernels and converts its weights. After
one such start this tool packages both, so later starts load them instead:

    kernel-cache manifest | build-image | check-log    compiled kernels, baked into the image
    tt-weights upload | download | original-subset     converted weights, in a Hugging Face repo
    release-hash                                       one JSON file that names both

Run it from the repo root: python3 scripts/prebuilt_artifacts.py <command> --help
The tt-weights commands need huggingface_hub and take the token from HF_TOKEN.
See tt-media-server/docs/minimax-h3-prebuilt-artifacts.md.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from workflows.kernel_cache import (
    KERNEL_CACHE_CONTAINER_DIR,
    KERNEL_CACHE_ENV,
    KERNEL_CACHE_MANIFEST_CONTAINER_PATH,
)

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
# tt-metal logs "compiled <kernel>/<hash>/ in <N> ms" per compiled kernel; a cache hit logs nothing.
COMPILE_LINE = re.compile(r"compiled (\S+/[^/\s]+)/ in \d+ ms")
# tt-metal writes a cache file as <stem>.<random id>_<pid><ext> and renames it into place.
KERNEL_CACHE_TEMP_FILE = re.compile(r"\.\d+_\d+(\.|$)")

TT_WEIGHTS_FOLDER = "tt_dit_cache"
TT_WEIGHTS_MANIFEST = "tt_weights_manifest.json"
# tt_dit loads a cache directory only when this (empty) file is in it.
CACHE_MARKER = "cache_dict.json"
TT_WEIGHTS_STAGING = ".tt_weights_download"

# Checkpoint folder of the transformer each MiniMax-H3 variant loads.
TRANSFORMER_FOLDER = {"fl2va": "transformer", "ref2va": "transformer_ref"}
VISION_TOWER_PREFIX = "model.visual."
# Matches the local_subdir of hf_extra_files in the MiniMax-H3 model specs.
LORA_FOLDER = "lightx2v"


class CliError(Exception):
    """A failure reported on stderr with exit code 1."""


def step(message: str) -> None:
    print(message, flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gigabytes(size: int) -> str:
    return f"{size / 1e9:.2f} GB"


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def docker(docker_run, args: list[str], capture: bool = False) -> str:
    command = ["docker", *args]
    result = docker_run(command, capture_output=capture, text=True)
    if result.returncode != 0:
        action = " ".join(itertools.takewhile(lambda arg: arg[0] != "-", command))
        detail = f": {result.stderr.strip()}" if capture and result.stderr else ""
        raise CliError(f"{action} failed{detail}")
    return result.stdout.strip() if capture else ""


def kernel_cache_files(cache_dir: Path):
    """Yield the cache files an image needs: no compiler logs, no temp files."""
    for root, dirs, names in os.walk(cache_dir):
        dirs.sort()
        for name in sorted(names):
            if name.endswith(".log") or KERNEL_CACHE_TEMP_FILE.search(name):
                continue
            yield Path(root) / name


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
        if line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def spec_env_values(path: Path) -> dict:
    """env_vars of a runtime model spec JSON, as run.py passes them with -e."""
    spec = read_json(path)
    spec = spec.get("runtime_model_spec", spec)
    env_vars = {key: str(value) for key, value in (spec.get("env_vars") or {}).items()}
    return {key: value for key, value in env_vars.items() if value}


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
    files = list(kernel_cache_files(cache_dir))
    size = sum(path.stat().st_size for path in files)
    step(
        f"scan: {len(build_keys)} build key(s), {kernel_hash_dirs} kernel hash folders, "
        f"{len(files)} files, {gigabytes(size)} in {cache_dir}"
    )

    compiled = len(compiled_kernels(server_log))
    step(f"log: {compiled} kernels compiled in {server_log}")

    layers = [image_env(docker_run, base_image)]
    if env_file:
        layers.append(env_file_values(env_file))
    if spec_json:
        layers.append(spec_env_values(spec_json))
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


def build_kernel_cache_image(
    docker_run, base_image, cache_dir, manifest_path, tag, push
) -> None:
    manifest = read_json(manifest_path)
    if manifest["base_image"] != base_image:
        raise CliError(
            f"the manifest was made for base image {manifest['base_image']}, "
            f"not {base_image}"
        )
    with tempfile.TemporaryDirectory(prefix="kernel-cache-image-") as context:
        context = Path(context)
        files = size = 0
        for source in kernel_cache_files(cache_dir):
            target = context / "kernel_cache" / source.relative_to(cache_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            files += 1
            size += target.stat().st_size
        shutil.copyfile(manifest_path, context / "kernel_cache_manifest.json")
        dockerfile = context / "Dockerfile"
        dockerfile.write_text(
            render_kernel_cache_dockerfile(manifest["build_key_env"]), encoding="utf-8"
        )
        step(f"stage: {files} files, {gigabytes(size)} in {context}")

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


def scan_tt_weights(cache_dir: Path) -> list[dict]:
    """One entry per directory that holds files: path, file count and bytes without the marker.

    Raises when a directory is not a finished conversion.
    """
    directories, problems = [], []
    for root, dirs, names in os.walk(cache_dir):
        dirs.sort()
        if not names:
            continue
        path = Path(root)
        relative = path.relative_to(cache_dir).as_posix()
        temp_files = fnmatch.filter(names, "*.tmp.*")
        if temp_files:
            problems.append(f"{relative}: {len(temp_files)} temp file(s)")
        if CACHE_MARKER not in names:
            problems.append(f"{relative}: no {CACHE_MARKER}")
        tensors = [name for name in names if name != CACHE_MARKER]
        directories.append(
            {
                "path": relative,
                "files": len(tensors),
                "bytes": sum((path / name).stat().st_size for name in tensors),
            }
        )
    if problems:
        raise CliError(
            "refusing to upload an unfinished conversion:\n  " + "\n  ".join(problems)
        )
    if not directories:
        raise CliError(f"no converted weights under {cache_dir}")
    return directories


def upload_tt_weights(api, cache_dir, repo, revision, tag, meta, manifest_out) -> None:
    """Make tt_dit_cache/ at the revision an exact copy of cache_dir.

    Files of an earlier upload that are no longer in cache_dir are removed from
    the revision; earlier versions stay reachable through their tags.
    """
    directories = scan_tt_weights(cache_dir)
    files = sum(entry["files"] for entry in directories)
    size = sum(entry["bytes"] for entry in directories)
    step(
        f"scan: {len(directories)} directories, {files} files, {gigabytes(size)} "
        f"in {cache_dir}"
    )

    if api.repo_exists(repo_id=repo):
        if tag and tag in [ref.name for ref in api.list_repo_refs(repo_id=repo).tags]:
            raise CliError(f"tag {tag} already exists in {repo}; tags are not moved")
        step(f"repo: {repo} exists")
    else:
        api.create_repo(repo_id=repo, private=True)
        step(f"repo: created {repo} as private")
    api.create_branch(repo_id=repo, branch=revision, exist_ok=True)

    api.upload_folder(
        repo_id=repo,
        folder_path=str(cache_dir),
        path_in_repo=TT_WEIGHTS_FOLDER,
        revision=revision,
        commit_message=f"Upload {TT_WEIGHTS_FOLDER}",
        ignore_patterns=[CACHE_MARKER, f"*/{CACHE_MARKER}"],
        delete_patterns=["*"],
    )
    step(f"upload: {TT_WEIGHTS_FOLDER}/ to {repo} at {revision}")

    manifest = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "repo": repo,
        "revision": revision,
        "tag": tag,
        "folder": TT_WEIGHTS_FOLDER,
        "marker": CACHE_MARKER,
        "meta": meta,
        "directories": directories,
        "files": files,
        "bytes": size,
    }
    commit = api.upload_file(
        path_or_fileobj=(json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
        path_in_repo=TT_WEIGHTS_MANIFEST,
        repo_id=repo,
        revision=revision,
        commit_message=f"Upload {TT_WEIGHTS_MANIFEST}",
    ).oid
    step(f"commit: {commit}")
    if tag:
        api.create_tag(repo_id=repo, tag=tag, revision=commit)
        step(f"tag: {tag}")

    write_json(manifest_out, {**manifest, "commit": commit})
    step(f"manifest: {manifest_out}")


def download_tt_weights(api, repo, revision, dest) -> None:
    commit = api.model_info(repo_id=repo, revision=revision).sha
    step(f"revision: {revision} is commit {commit}")

    staging = dest / TT_WEIGHTS_STAGING
    manifest = read_json(
        api.hf_hub_download(
            repo_id=repo,
            filename=TT_WEIGHTS_MANIFEST,
            revision=commit,
            local_dir=str(staging),
        )
    )
    directories = manifest["directories"]
    for entry in directories:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise CliError(f"unsafe directory in the manifest: {entry['path']}")
    step(
        f"manifest: {len(directories)} directories, {manifest['files']} files, "
        f"{gigabytes(manifest['bytes'])}"
    )

    api.snapshot_download(
        repo_id=repo,
        revision=commit,
        allow_patterns=[f"{TT_WEIGHTS_FOLDER}/*"],
        local_dir=str(staging),
    )
    step(f"download: {TT_WEIGHTS_FOLDER}/ of {repo} into {dest}")

    mismatches = []
    for entry in directories:
        source = staging / TT_WEIGHTS_FOLDER / entry["path"]
        target = dest / entry["path"]
        target.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            for item in source.iterdir():
                if item.is_file():
                    os.replace(item, target / item.name)
        tensors = [
            item
            for item in target.iterdir()
            if item.is_file() and item.name != CACHE_MARKER
        ]
        size = sum(item.stat().st_size for item in tensors)
        if (len(tensors), size) != (entry["files"], entry["bytes"]):
            (target / CACHE_MARKER).unlink(missing_ok=True)
            mismatches.append(
                f"{entry['path']}: {len(tensors)} files, {size} bytes; the manifest "
                f"says {entry['files']} files, {entry['bytes']} bytes"
            )
    if mismatches:
        raise CliError(
            "the download does not match the manifest; no directory was marked "
            "complete:\n  " + "\n  ".join(mismatches)
        )
    step(f"verify: {len(directories)} directories match the manifest")

    for entry in directories:
        (dest / entry["path"] / CACHE_MARKER).touch()
    shutil.rmtree(staging)
    step(f"mark: {len(directories)} directories complete")


def download_original_subset(api, repo, variant, dest, lora) -> None:
    """Fetch what a start with a complete converted cache still reads from the checkpoint."""
    transformer = TRANSFORMER_FOLDER[variant]
    api.snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "audio_scheduler/*",
            "processor/*",
            "tokenizer/*",
            f"{transformer}/config.json",
            "audio_vae/config.json",
            "vae/*",
            "text_encoder/*",
        ],
        ignore_patterns=["text_encoder/*.safetensors"],
    )
    step(f"download: configs, tokenizer and vae of {repo} into {dest}")

    weight_map = read_json(dest / "text_encoder" / "model.safetensors.index.json")[
        "weight_map"
    ]
    shards = sorted(
        {
            shard
            for name, shard in weight_map.items()
            if name.startswith(VISION_TOWER_PREFIX)
        }
    )
    if not shards:
        raise CliError(f"no {VISION_TOWER_PREFIX}* weights in the text encoder index")
    api.snapshot_download(
        repo_id=repo,
        local_dir=str(dest),
        allow_patterns=[f"text_encoder/{shard}" for shard in shards],
    )
    step(f"download: vision tower shard(s) {', '.join(shards)}")

    folders = (transformer, "text_encoder", "vae", "audio_vae")
    for folder in folders:
        (dest / folder).mkdir(parents=True, exist_ok=True)
    step(f"folders: {', '.join(folders)}")

    if lora:
        lora_repo, _, filename = lora.partition(":")
        if not lora_repo or not filename:
            raise CliError("--lora takes REPO:FILENAME")
        api.hf_hub_download(
            repo_id=lora_repo, filename=filename, local_dir=str(dest / LORA_FOLDER)
        )
        step(f"lora: {LORA_FOLDER}/{filename}")


def write_release_hash(args) -> None:
    kernel = read_json(args.kernel_manifest)
    if kernel["base_image"] != args.base_image:
        raise CliError(
            f"the kernel manifest was made for base image {kernel['base_image']}, "
            f"not {args.base_image}"
        )
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
        weights = read_json(args.weights_manifest)
        tt_weights = {
            key: weights[key]
            for key in ("repo", "revision", "tag", "commit", "files", "bytes", "meta")
        }

    identity = {
        "model": args.model,
        "device": args.device,
        "tt_metal_sha": args.tt_metal_sha,
        "server_sha": args.server_sha,
        "image_digest": args.image_digest,
        "kernel_cache_env": kernel_cache_env,
        "tt_weights": tt_weights
        and {"repo": tt_weights["repo"], "commit": tt_weights["commit"]},
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
    step(f"release-hash: {release_hash}")
    step(f"release: {args.out}")


def parse_meta(pairs: list[str]) -> dict:
    meta = {}
    for pair in pairs:
        key, separator, value = pair.partition("=")
        if not key or not separator:
            raise CliError(f"--meta takes K=V, got {pair!r}")
        meta[key] = value
    return meta


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
    manifest.add_argument("--cache-dir", type=Path, required=True)
    manifest.add_argument("--server-log", type=Path, required=True)
    manifest.add_argument("--base-image", required=True)
    manifest.add_argument("--env-file", type=Path)
    manifest.add_argument("--spec-json", type=Path)
    manifest.add_argument("--out", type=Path, required=True)

    build_image = kernel_cache.add_parser(
        "build-image", help="add a captured kernel cache to the image it came from"
    )
    build_image.add_argument("--base-image", required=True)
    build_image.add_argument("--cache-dir", type=Path, required=True)
    build_image.add_argument("--manifest", type=Path, required=True)
    build_image.add_argument("--tag", required=True)
    build_image.add_argument("--push", action="store_true")

    check = kernel_cache.add_parser(
        "check-log", help="fail when a server start compiled kernels"
    )
    check.add_argument("--server-log", type=Path, required=True)
    check.add_argument("--max-compiled", type=int, default=0)

    tt_weights = commands.add_parser(
        "tt-weights", help="converted weights, in a Hugging Face repo"
    ).add_subparsers(dest="action", required=True)

    upload = tt_weights.add_parser("upload", help="publish a converted weights cache")
    upload.add_argument("--cache-dir", type=Path, required=True)
    upload.add_argument("--repo", required=True)
    upload.add_argument("--revision", required=True)
    upload.add_argument("--tag")
    upload.add_argument("--meta", nargs="+", action="extend", default=[], metavar="K=V")
    upload.add_argument("--manifest-out", type=Path, required=True)

    download = tt_weights.add_parser(
        "download", help="fetch and verify a converted weights cache"
    )
    download.add_argument("--repo", required=True)
    download.add_argument("--revision", required=True)
    download.add_argument("--dest", type=Path, required=True)

    subset = tt_weights.add_parser(
        "original-subset",
        help="fetch the part of the original checkpoint a cached start still reads",
    )
    subset.add_argument("--repo", required=True)
    subset.add_argument("--variant", choices=sorted(TRANSFORMER_FOLDER), required=True)
    subset.add_argument("--dest", type=Path, required=True)
    subset.add_argument("--lora", metavar="REPO:FILENAME")

    release = commands.add_parser(
        "release-hash", help="write the file handed to the deployment team"
    )
    release.add_argument("--base-image", required=True)
    release.add_argument("--image", required=True)
    release.add_argument("--image-digest", required=True)
    release.add_argument("--kernel-manifest", type=Path, required=True)
    release.add_argument("--weights-manifest", type=Path)
    release.add_argument("--tt-metal-sha", required=True)
    release.add_argument("--server-sha", required=True)
    release.add_argument("--model", required=True)
    release.add_argument("--device", required=True)
    release.add_argument("--warm-ready-seconds", type=float)
    release.add_argument("--run-url")
    release.add_argument("--out", type=Path, required=True)
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
            upload_tt_weights(
                api,
                args.cache_dir,
                args.repo,
                args.revision,
                args.tag,
                parse_meta(args.meta),
                args.manifest_out,
            )
        elif args.action == "download":
            download_tt_weights(api, args.repo, args.revision, args.dest)
        else:
            download_original_subset(api, args.repo, args.variant, args.dest, args.lora)
    return 0


def main(argv=None, docker_run=subprocess.run, hf_api=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args, docker_run, hf_api)
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
