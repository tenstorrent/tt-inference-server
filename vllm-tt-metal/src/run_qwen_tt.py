#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Launch tt-lab's prebuilt ``tt-lab serve`` for Qwen3.8-27B (impl tt_lab_qwen27b).

The device spec makes this script the container's entrypoint (docker_entrypoint), so any
image with a python3 can serve the model. Standard library only. The package (the
QWEN_TT_PACKAGE folder in the weights repo, or a host package mounted read-only) holds
``bin/tt-lab`` and ``runtime/`` (its own loader and C/C++ libraries, so the image's
libraries are irrelevant), and ``SHA256SUMS``. A package without exec bits (a Hugging Face
download) is verified against its SHA256SUMS and copied to a temporary directory to run.

Environment (run_docker_server.py sets MODEL, DEVICE, CACHE_ROOT and the spec's env_vars):
  QWEN_TT_PACKAGE    the package's folder in the weights directory (default tt-lab)
  QWEN_TT_HOME       the package (default QWEN_TT_PACKAGE in the weights directory, else
                     /home/container_app_user/tt-lab)
  DEVICE             p150x8 (LoudBox) or blackhole_galaxy (one 8-chip group); bh_galaxy too.
  QWEN_TT_TTQ, QWEN_TT_VOCAB  the weights: absolute paths, or names in the weights
                     directory (MODEL_WEIGHTS_DIR, else $CACHE_ROOT/weights/$MODEL where
                     TIS downloads the weights repo), or in an HF snapshot under HF_HOME.
                     QWEN_TT_VOCAB is a vocabulary-only GGUF: serve reads only the
                     tokenizer, chat template and architecture keys from it.
  VLLM_API_KEY / API_KEY / JWT_SECRET / NO_AUTH  the bearer key, in that order: TIS's
                     clients send a MEDIA server the literal VLLM_API_KEY or API_KEY. It
                     reaches serve in the TT_LAB_API_KEY environment variable, never argv.
  SERVICE_PORT (8000), SERVED_MODEL_NAME ($MODEL), QWEN_TT_MAX_TOKENS (32768),
  QWEN_TT_USERS (8), QWEN_TT_CONTEXT (131072), QWEN_TT_HOST (0.0.0.0),
  QWEN_TT_EXTRA_ARGS (appended to serve's arguments, split on spaces).
The vLLM image's arguments ``--model M --tt-device D --service-port P --no-auth`` are
accepted too, and ``tt-lab ARGS...`` runs the binary.

Exit status is tt-lab's; 64 is a bad setup here.
"""

import base64
import glob
import hashlib
import hmac
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_HOME = "/home/container_app_user/tt-lab"
SETUP_FAILED = 64

# DEVICE -> (chips, extra serve arguments, environment defaults)
DEVICES = {
    "p150x8": (8, [], {}),
    # One 8-chip group of the Galaxy's 32 chips.
    "blackhole_galaxy": (8, [], {}),
}
DEVICES["bh_galaxy"] = DEVICES["blackhole_galaxy"]

# The fixed JWT header and payload TIS's clients sign (pyjwt's compact JSON).
JWT_HEADER = b'{"alg":"HS256","typ":"JWT"}'
JWT_PAYLOAD = b'{"team_id":"tenstorrent","token_id":"debug-test"}'


class SetupError(Exception):
    pass


def log(message):
    print(f"run_qwen_tt: {message}", file=sys.stderr, flush=True)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def tis_jwt(secret: str) -> str:
    """HS256 JWT of TIS's fixed payload, byte-identical to pyjwt's."""
    signing_input = f"{_b64url(JWT_HEADER)}.{_b64url(JWT_PAYLOAD)}"
    signature = hmac.new(
        secret.encode(), signing_input.encode(), hashlib.sha256
    ).digest()
    return f"{signing_input}.{_b64url(signature)}"


def _truthy(value) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes")


def resolve_api_key(env):
    """(key or None, source): VLLM_API_KEY, API_KEY, the JWT of JWT_SECRET; or NO_AUTH.

    TIS's clients send a MEDIA server the literal VLLM_API_KEY or API_KEY (Shield sets
    API_KEY), and a JWT of JWT_SECRET only to vLLM servers.
    """
    if _truthy(env.get("NO_AUTH")):
        return None, "NO_AUTH"
    if env.get("VLLM_API_KEY"):
        return env["VLLM_API_KEY"], "VLLM_API_KEY"
    if env.get("API_KEY"):
        return env["API_KEY"], "API_KEY"
    if env.get("JWT_SECRET"):
        return tis_jwt(env["JWT_SECRET"]), "JWT_SECRET"
    raise SetupError(
        "set JWT_SECRET, VLLM_API_KEY or API_KEY, or NO_AUTH=1 to serve without a key"
    )


def weights_dir(env):
    """MODEL_WEIGHTS_DIR, else $CACHE_ROOT/weights/$MODEL (TIS's download of the weights
    repo into the persistent volume) when it exists, else None."""
    if env.get("MODEL_WEIGHTS_DIR"):
        return env["MODEL_WEIGHTS_DIR"]
    if env.get("CACHE_ROOT") and env.get("MODEL"):
        path = os.path.join(env["CACHE_ROOT"], "weights", env["MODEL"])
        if os.path.isdir(path):
            return path
    return None


def _weight_file(path):
    """path if it is a file, its split TTQ's manifest if that is, else None."""
    if os.path.isfile(path):
        return path
    if os.path.isfile(f"{path}.manifest"):
        return f"{path}.manifest"
    return None


def check_split_ttq(manifest):
    """Raise SetupError unless every part a split TTQ's manifest lists is present at full size.

    A missing or partial download then fails here, before tt-lab touches the chips (and so
    before it could mark them dirty).
    """
    folder = os.path.dirname(manifest)
    problems = []
    with open(manifest) as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 4 or fields[0] != "part":
                continue
            name, size = fields[1], int(fields[3])
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                problems.append(f"{name} is missing")
            elif os.path.getsize(path) != size:
                problems.append(f"{name} has {os.path.getsize(path)} of {size} bytes")
    if problems:
        raise SetupError(f"split TTQ {manifest} is incomplete: {'; '.join(problems)}")


def find_weight(name, env):
    """The weight file's path, or None: absolute, in the weights directory, or in HF_HOME."""
    if name.startswith("/"):
        return _weight_file(name)
    directory = weights_dir(env)
    if directory:
        return _weight_file(os.path.join(directory, name))
    hf_home = env.get("HF_HOME", "/nonexistent")
    for snapshot in sorted(glob.glob(f"{hf_home}/hub/models--*/snapshots/*")):
        path = _weight_file(os.path.join(snapshot, name))
        if path:
            return path
    return None


def package_home(env) -> Path:
    """QWEN_TT_HOME, else QWEN_TT_PACKAGE (default tt-lab) in the weights directory when present,
    else the default.

    A weights repo keeps each build in its own folder (tt-lab-<commit>) so that a cached
    download, which is complete while the old folder is there, still fetches a new build.
    """
    if env.get("QWEN_TT_HOME"):
        return Path(env["QWEN_TT_HOME"])
    directory = weights_dir(env)
    name = env.get("QWEN_TT_PACKAGE") or "tt-lab"
    if directory and (Path(directory) / name / "bin" / "tt-lab").is_file():
        return Path(directory) / name
    return Path(DEFAULT_HOME)


def runnable_home(home: Path, staging_root=None) -> Path:
    """home if its loader and binaries can execute; else a verified, executable copy.

    A Hugging Face download keeps no exec bits and the weights mount may be read-only, so
    bin/ and runtime/ are checked against SHA256SUMS and copied to a temporary directory.
    """
    executables = [home / "runtime/ld-linux-x86-64.so.2", home / "bin/tt-lab"]
    if all(os.access(path, os.X_OK) for path in executables):
        return home
    if not (home / "SHA256SUMS").is_file():
        raise SetupError(
            f"{home} is not executable and has no SHA256SUMS to verify a copy against"
        )
    verify_package(home)
    staged = Path(tempfile.mkdtemp(prefix="tt-lab-", dir=staging_root))
    for name in ("bin", "runtime"):
        shutil.copytree(home / name, staged / name)
        for path in (staged / name).iterdir():
            path.chmod(0o755)
    for name in ("SOURCE", "SHA256SUMS"):
        if (home / name).is_file():
            shutil.copy(home / name, staged / name)
    log(f"{home} is not executable; running a verified copy in {staged}")
    return staged


def verify_package(home: Path):
    """Check the package's files against its SHA256SUMS, when it has one."""
    sums = home / "SHA256SUMS"
    if not sums.is_file():
        log(f"{home} has no SHA256SUMS; not verifying the package")
        return
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = home / name.lstrip("*")
        if not path.is_file():
            raise SetupError(f"package file {path} (listed in SHA256SUMS) is missing")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise SetupError(f"package file {path} has sha256 {actual}, not {digest}")


def serve_command(env, usage, chips_present):
    """(serve arguments, environment updates) for tt-lab, from the environment.

    ``usage`` is tt-lab's usage text (its features), ``chips_present`` the number of
    /dev/tenstorrent chips.
    """
    device = (env.get("DEVICE") or "").lower()
    if not device:
        raise SetupError("DEVICE is not set (p150x8 or blackhole_galaxy)")
    if device not in DEVICES:
        raise SetupError(
            f"DEVICE={device} is not supported (p150x8 or blackhole_galaxy)"
        )
    chips, device_args, device_env = DEVICES[device]
    if chips_present < chips:
        raise SetupError(
            f"DEVICE={device} needs {chips} chips in /dev/tenstorrent, found {chips_present}"
        )
    updates = {k: env.get(k) or v for k, v in device_env.items()}

    ttq_name = env.get("QWEN_TT_TTQ") or "Qwen3.8-27B-bfp4-gptq-bfp8ah.ttq"
    vocab_name = env.get("QWEN_TT_VOCAB") or "Qwen3.8-27B-vocab.gguf"
    ttq, vocab = find_weight(ttq_name, env), find_weight(vocab_name, env)
    where = env.get("MODEL_WEIGHTS_DIR") or f"{env.get('HF_HOME', '<unset>')}/hub"
    if not ttq:
        raise SetupError(f"weights: {ttq_name} not found (in {where}); set QWEN_TT_TTQ")
    if not vocab:
        raise SetupError(
            f"weights: {vocab_name} not found (in {where}); set QWEN_TT_VOCAB"
        )

    # TIS publishes the container's port, so serve must listen beyond loopback.
    if "--host" not in usage:
        raise SetupError(
            "this tt-lab has no --host, so it listens on 127.0.0.1 only and the published "
            "port cannot reach it; use a tt-lab with --host"
        )
    args = ["--host", env.get("QWEN_TT_HOST") or "0.0.0.0"]

    key, source = resolve_api_key(env)
    if key is None:
        log("NO_AUTH set, serving without an API key")
        updates["TT_LAB_API_KEY"] = None
    else:
        log(f"API key from {source}")
        if "TT_LAB_API_KEY" not in usage:
            raise SetupError("API key set but this tt-lab does not read TT_LAB_API_KEY")
        updates["TT_LAB_API_KEY"] = key

    model = env.get("MODEL") or "Qwen3.8-27B"
    if "/" not in model:
        model = f"Qwen/{model}"
    serve = [
        "serve",
        "-m", vocab,
        "--ttq", ttq,
        "--device",
        "--tiles", "64",
        "--prefill-chunk", "2048",
        "--users", env.get("QWEN_TT_USERS") or "8",
        "--context", env.get("QWEN_TT_CONTEXT") or "131072",
        "-n", env.get("QWEN_TT_MAX_TOKENS") or "32768",
        "--port", env.get("SERVICE_PORT") or "8000",
        "--served-model-name", env.get("SERVED_MODEL_NAME") or model,
    ]  # fmt: skip
    extra = shlex.split(env.get("QWEN_TT_EXTRA_ARGS", ""))
    return serve + args + device_args + extra, updates


def parse_vllm_args(argv, env):
    """Accept the vLLM image's arguments, mapped onto the environment."""
    flags = {
        "--model": "MODEL",
        "--tt-device": "DEVICE",
        "--service-port": "SERVICE_PORT",
    }
    i = 0
    while i < len(argv):
        if argv[i] in flags and i + 1 < len(argv):
            env[flags[argv[i]]] = argv[i + 1]
            i += 2
        elif argv[i] == "--no-auth":
            env["NO_AUTH"] = "1"
            i += 1
        else:
            raise SetupError(
                f"unknown argument {argv[i]} (takes --model, --tt-device, --service-port, "
                f"--no-auth)"
            )


def run_serve(command, env):
    """Run tt-lab as a child, forwarding each SIGTERM/SIGINT, and return its status."""
    children = []

    def forward(signum, _frame):
        for child in children:
            child.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward)
    children.append(subprocess.Popen(command, env=env))
    status = children[0].wait()
    # Killed by a signal: report it as a shell would (128 + the signal number).
    return 128 - status if status < 0 else status


def device_report(chips, sys_root="/sys"):
    """One line on the host DMA setup: each chip's IOMMU group type and the free 1 GiB pages.

    tt-lab pins ordinary host memory for NOC DMA, which needs the IOMMU to translate (group
    type DMA or DMA-FQ); a chip in an identity (passthrough) group fails to pin.
    """
    types = {}
    for chip in chips:
        number = chip.rsplit("/", 1)[-1]
        group = Path(
            sys_root,
            "class",
            "tenstorrent",
            f"tenstorrent!{number}",
            "device",
            "iommu_group",
        )
        try:
            kind = (group / "type").read_text().strip()
        except OSError:
            kind = "unknown"
        types.setdefault(kind, []).append(int(number))
    pages = Path(sys_root, "kernel", "mm", "hugepages", "hugepages-1048576kB")
    try:
        huge = f"{(pages / 'free_hugepages').read_text().strip()} of {(pages / 'nr_hugepages').read_text().strip()} free"
    except OSError:
        huge = "none"
    groups = "; ".join(
        f"{kind}: {sorted(numbers)}" for kind, numbers in sorted(types.items())
    )
    return f"IOMMU groups by type: {groups or 'no chips'}; 1 GiB hugepages {huge}"


def main(argv=None, environ=None):
    argv = sys.argv[1:] if argv is None else argv
    env = dict(os.environ if environ is None else environ)
    try:
        passthrough = bool(argv) and argv[0] in ("tt-lab", "test-tt-kernels")
        if not passthrough:
            parse_vllm_args(argv, env)
        home = package_home(env)
        if not (home / "bin/tt-lab").is_file():
            raise SetupError(
                f"no tt-lab in {home}/bin (put the package at QWEN_TT_HOME or tt-lab/ in "
                f"the weights directory)"
            )
        verified = home
        home = runnable_home(home)
        loader = [
            str(home / "runtime/ld-linux-x86-64.so.2"),
            "--library-path",
            str(home / "runtime"),
        ]
        if passthrough:
            os.execv(loader[0], loader + [str(home / "bin" / argv[0])] + argv[1:])
        tt_lab = loader + [str(home / "bin/tt-lab")]
        if verified == home:
            verify_package(home)
        if (home / "SOURCE").is_file():
            sys.stderr.write((home / "SOURCE").read_text())
        usage = subprocess.run(
            tt_lab, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ).stdout
        chip_paths = sorted(glob.glob("/dev/tenstorrent/[0-9]*"))
        chips = len(chip_paths)
        log(device_report(chip_paths))
        serve, updates = serve_command(env, usage, chips)
    except SetupError as error:
        log(str(error))
        return SETUP_FAILED
    for name, value in updates.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    log(f"DEVICE={env.get('DEVICE')}")
    log(f"tt-lab {' '.join(serve)}")
    status = run_serve(tt_lab + serve, env)
    log(f"tt-lab exited {status}")
    return status


if __name__ == "__main__":
    sys.exit(main())
