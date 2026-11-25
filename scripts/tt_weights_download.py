#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Download the converted weights of one model variant from a Hugging Face repo.

scripts/prebuilt_artifacts.py tt-weights upload publishes them. This script
fetches what the manifest of one variant lists, verifies it and marks the cache
directories complete, so a server start loads them instead of converting:

    python3 tt_weights_download.py --repo <org>/<name> --revision <tag or commit> \
        --variant <variant> --dest-cache <TT_DIT_CACHE_DIR> --dest-original <checkpoint dir>

A download needs huggingface_hub and takes the token from HF_TOKEN. A run on a
destination that the same revision completed needs neither: it checks the
destination against the record saved there, converted files by name and size
plus the completeness marker, original files by sha256.

The kernel-cache image carries this file alone, so it imports nothing of the
repo, and huggingface_hub only for a download. scripts/prebuilt_artifacts.py
imports the layout of the weights repo and the Hub helpers from here.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path, PurePosixPath

# One private repo holds every variant of a model family, side by side:
#   tt_dit_cache/<tree as under TT_DIT_CACHE_DIR>   converted weights, shared directories stored once
#   original_subset/<paths as in the checkpoint>    what a cached start still reads from the original
#   manifests/<variant>.json                        what one variant needs of the two
TT_WEIGHTS_FOLDER = "tt_dit_cache"
ORIGINAL_SUBSET_FOLDER = "original_subset"
MANIFEST_FOLDER = "manifests"
VARIANT_SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
# tt_dit loads a cache directory only when this (empty) file is in it.
CACHE_MARKER = "cache_dict.json"
TT_WEIGHTS_STAGING = ".tt_weights_download"
# A completed download saves <variant>.json in this folder of the cache
# destination: the repo, the revision, its commit, the manifest and the file
# sizes of each cache directory. A run for that revision checks the destination
# against it instead of asking the Hub.
TT_WEIGHTS_RECORDS = ".tt_weights_complete"
# Layout version of the manifests and the release file; a reader refuses another.
SCHEMA_VERSION = 1
WEIGHTS_MANIFEST_FIELDS = (
    "variant",
    "cache_directories",
    "original_files",
    "original_folders",
)


class CliError(Exception):
    """A failure reported on stderr with exit code 1."""


def step(message: str) -> None:
    print(message, flush=True)


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
    if data.get("schema_version") != SCHEMA_VERSION:
        raise CliError(
            f"{source} has schema_version {data.get('schema_version')}; this "
            f"command reads {SCHEMA_VERSION}"
        )
    return {name: data[name] for name in names}


def hugging_face_api():
    """Hub client; huggingface_hub reads the token from HF_TOKEN itself."""
    if not os.environ.get("HF_TOKEN"):
        raise CliError("HF_TOKEN is not set")
    try:
        from huggingface_hub import HfApi
    except ImportError as error:
        raise CliError("huggingface_hub is not installed") from error
    return HfApi()


def file_hash(path: Path, digest=None) -> str:
    """sha256 of a file, or its hash in the digest given."""
    digest = digest or hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def hub_reason(repo, revision, variant, dest_cache, dest_original) -> str | None:
    """Why a run needs the Hub, or None when the destination is complete by its record.

    The record of the variant has to name the repo and the revision or its
    commit. The destination is then checked against it without the Hub: each
    cache directory marked and holding the recorded file names and sizes, each
    original file with its sha256, each original folder present.
    """
    path = dest_cache / TT_WEIGHTS_RECORDS / f"{variant}.json"
    if not path.is_file():
        return f"{dest_cache} has no record of a complete download of {variant}"
    with exclusive(dest_cache):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["schema_version"] != SCHEMA_VERSION:
                raise ValueError("schema_version")
            commit, manifest = record["commit"], record["manifest"]
            if (record["repo"], record["variant"]) != (repo, variant):
                return f"{path} records {record['variant']} of {record['repo']}"
            if revision not in (record["revision"], commit):
                return (
                    f"{path} records revision {record['revision']} "
                    f"(commit {commit}), not {revision}"
                )
            for entry in manifest["cache_directories"]:
                directory = dest_cache / entry["path"]
                if not (directory / CACHE_MARKER).is_file():
                    return f"{directory} has no {CACHE_MARKER}"
                if tensor_sizes(directory) != record["cache_files"][entry["path"]]:
                    return (
                        f"{directory} does not hold the recorded file names and sizes"
                    )
            for entry in manifest["original_files"]:
                if not is_listed_file(dest_original / entry["path"], entry):
                    return (
                        f"{dest_original / entry['path']} is missing or not the "
                        "recorded file"
                    )
            for folder in manifest["original_folders"]:
                if not (dest_original / folder).is_dir():
                    return f"the folder {dest_original / folder} is missing"
        except (KeyError, TypeError, ValueError):
            return f"{path} is not a record this script reads"
    step(
        f"complete: {dest_cache} holds {variant} of {repo} at commit {commit}; "
        f"{len(manifest['cache_directories'])} cache directories and "
        f"{len(manifest['original_files'])} original files re-checked against the "
        "saved manifest; the Hub was not contacted"
    )
    return None


def download_tt_weights(
    api, repo, revision, variant, dest_cache, dest_original
) -> None:
    """Fetch what manifests/<variant>.json of a revision lists and mark what was verified.

    What the destination already holds with the content of that revision is
    kept as it is, marker included: it is not fetched again, and a variant that
    shares it keeps working if this download fails. A cache directory with
    files the revision does not hold is refused before anything is changed or
    fetched. The markers of the other cache directories are removed first, so a
    download that stops part way leaves none of them looking complete. The
    files are checked in a staging folder before any of them is moved into
    place. A failed run keeps that folder, and the next run for the same commit
    continues from it. A run that fetched a cache directory removes the records
    of every variant before it moves the files into place: the other variants
    are checked against the Hub on their next run. A completed run saves the
    record that hub_reason reads.
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
    remote = {
        entry["path"]: published_directory(files, entry["path"]) for entry in listed
    }
    directories = [
        entry
        for entry in listed
        if not holds_directory(dest_cache / entry["path"], remote[entry["path"]])
    ]
    unlisted = []
    for entry in directories:
        local = tensor_sizes(dest_cache / entry["path"])
        other = sorted(set(local) - set(remote[entry["path"]]))
        if other:
            unlisted.append(
                f"{entry['path']}: {len(other)} file(s), for example {other[0]}"
            )
    if unlisted:
        raise CliError(
            f"{dest_cache} holds files that {repo} does not hold at {revision}; "
            "nothing was changed or fetched. Remove these directories and run "
            "the download again:\n  " + "\n  ".join(unlisted)
        )
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
    # One pattern per file: "<directory>/*" would also match its sub-folders.
    patterns = [
        f"{TT_WEIGHTS_FOLDER}/{glob.escape(entry['path'])}/{glob.escape(name)}"
        for entry in directories
        for name in remote[entry["path"]]
    ] + [
        f"{ORIGINAL_SUBSET_FOLDER}/{glob.escape(entry['path'])}" for entry in originals
    ]
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
    for entry, source in zip(directories, staged):
        sizes = tensor_sizes(source)
        if (len(sizes), sum(sizes.values())) != (entry["files"], entry["bytes"]):
            mismatches.append(
                f"{entry['path']}: {len(sizes)} files, {sum(sizes.values())} bytes; "
                f"the manifest says {entry['files']} files, {entry['bytes']} bytes"
            )
    for entry in originals:
        if not is_listed_file(staging / ORIGINAL_SUBSET_FOLDER / entry["path"], entry):
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

    # A record vouches for file names and sizes only, and a directory can be
    # shared: no record outlives a run that moves a cache directory into place.
    records = dest_cache / TT_WEIGHTS_RECORDS
    if directories and records.exists():
        shutil.rmtree(records)
    for source, marker in zip(staged, markers):
        marker.parent.mkdir(parents=True, exist_ok=True)
        for name in tensor_sizes(source):
            os.replace(source / name, marker.parent / name)
    for entry in originals:
        target = dest_original / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(staging / ORIGINAL_SUBSET_FOLDER / entry["path"], target)
    for folder in folders:
        (dest_original / folder).mkdir(parents=True, exist_ok=True)
    # A kept marker can be a file this user may not write: only a missing one is made.
    for entry in listed:
        marker = dest_cache / entry["path"] / CACHE_MARKER
        if not marker.exists():
            marker.touch()
    shutil.rmtree(staging.parent, ignore_errors=True)
    step(f"mark: {len(listed)} directories complete in {dest_cache}")

    record = {
        "schema_version": SCHEMA_VERSION,
        "repo": repo,
        "variant": variant,
        "revision": revision,
        "commit": commit,
        "manifest": manifest,
        "cache_files": {
            path: {name: item.size for name, item in directory.items()}
            for path, directory in remote.items()
        },
    }
    saved = dest_cache / TT_WEIGHTS_RECORDS / f"{variant}.json"
    saved.parent.mkdir(exist_ok=True)
    saved.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    step(f"record: {saved} saved; a run for this revision checks against it")


def path_argument(value: str) -> Path:
    """A path option. An empty value would otherwise mean the current directory."""
    if not value.strip():
        raise argparse.ArgumentTypeError("the path is empty")
    return Path(value)


def variant_argument(value: str) -> str:
    """A variant slug. It names the manifest of the variant and starts its tag."""
    if not VARIANT_SLUG.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "a variant is a lowercase slug of letters and digits joined by "
            "single hyphens, in the form <model>-<task>"
        )
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and verify the converted weights one model variant "
        "needs. What the destination already holds with the same content is kept. "
        "Needs huggingface_hub and the token in HF_TOKEN, except on a destination "
        "that a run for the same revision completed: that one is checked against "
        "the record saved in it, without the Hub (converted files by name and "
        "size plus the completeness marker, original files by sha256)."
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Hugging Face repo that holds the converted weights, as <org>/<name>",
    )
    parser.add_argument(
        "--revision",
        required=True,
        help="tag or commit from the release file; a branch can be in the middle "
        "of an upload, and the Hub is not asked where it points once a run for it "
        "completed the destination",
    )
    parser.add_argument(
        "--variant",
        type=variant_argument,
        required=True,
        help="model variant to fetch; the repo holds its manifest as "
        f"{MANIFEST_FOLDER}/<variant>.json",
    )
    parser.add_argument(
        "--dest-cache",
        type=path_argument,
        required=True,
        help="directory the server uses as TT_DIT_CACHE_DIR; the converted "
        "weights go here",
    )
    parser.add_argument(
        "--dest-original",
        type=path_argument,
        required=True,
        help="the folder the server reads the original checkpoint from; the "
        "files it still reads from that checkpoint go here",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    target = (
        args.repo,
        args.revision,
        args.variant,
        args.dest_cache,
        args.dest_original,
    )
    try:
        reason = hub_reason(*target)
        if reason is None:
            return 0
        step(f"hub: needed, {reason}")
        api = hugging_face_api()
        with exclusive(args.dest_cache):
            download_tt_weights(api, *target)
        return 0
    except CliError as error:
        print(f"error: {error}", file=sys.stderr)
    except OSError as error:
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
