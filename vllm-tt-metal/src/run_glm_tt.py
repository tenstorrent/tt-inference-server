#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Launch tt-lab's prebuilt ``glm-tt serve`` (impl tt_lab_glm).

The device spec makes this script the container's entrypoint (docker_entrypoint), so any
image with a python3 can serve GLM-5.3-Flash. Standard library only: the image's venv is
not needed. The package (``glm-tt/`` in the weights repo, or a host package mounted
read-only) holds ``bin/glm-tt`` and ``runtime/`` (its own loader and C/C++ libraries, so
the image's libraries are irrelevant), optionally ``SOURCE`` and ``SHA256SUMS``. A
package without exec bits (a Hugging Face download) is verified against its SHA256SUMS
and copied to a temporary directory to run.

Environment (run_docker_server.py sets MODEL, DEVICE, CACHE_ROOT and the spec's env_vars):
  GLM_TT_PACKAGE     the package's folder in the weights directory (default glm-tt)
  GLM_TT_HOME        the package (default GLM_TT_PACKAGE in the weights directory, else
                     /home/container_app_user/glm-tt)
  DEVICE             p150x8 (LoudBox, one user) or blackhole_galaxy (Galaxy: --groups auto,
                     the 3-cube groups of the Galaxy's Ethernet link map, or --groups 4 with
                     a glm-tt that has no auto; TT_LAB_GLM_ROUND=auto, and
                     TT_LAB_GLM_GATHER=eth with a glm-tt that has --mtp); bh_galaxy too.
                     TT_LAB_ETH_GATHER=1 on both unless set.
  GLM_TT_MTP         the MTP sidecar TTQ (name or path, found as GLM_TT_TTQ is): serve --mtp.
  GLM_TT_FALLBACK    Galaxy: "pcie" (default) reruns serve on PCIe rounds and host gathers,
                     one user per group and no MTP or GLM_TT_EXTRA_ARGS, when glm-tt finds
                     no 3-cube groups (exit status 5); "none" exits 5 instead.
  GLM_TT_PROBE_ONLY  Galaxy, 1: print ``glm-tt groups`` (the link map, the cubes and the
                     cover; read-only) and exit with its status (0, or 5 for no cube groups)
                     without serving. Every Galaxy start prints it before serve.
  GLM_TT_TTQ, GLM_TT_VOCAB  the weights: absolute paths, or names in the weights
                     directory (MODEL_WEIGHTS_DIR, else $CACHE_ROOT/weights/$MODEL where
                     TIS downloads the weights repo), or in an HF snapshot under HF_HOME.
                     A TTQ split into parts is found by its NAME.manifest.
  VLLM_API_KEY / API_KEY / JWT_SECRET / NO_AUTH  the bearer key, in that order: TIS's
                     clients send a MEDIA server the literal VLLM_API_KEY or API_KEY
                     (workflow_module/command_factory.py _resolve_auth_token), and the
                     JWT of JWT_SECRET only to vLLM; NO_AUTH=1 for none.
  SERVICE_PORT (8000), SERVED_MODEL_NAME (zai-org/$MODEL), GLM_TT_MAX_TOKENS (163840),
  GLM_TT_HOST (0.0.0.0), GLM_TT_EXTRA_ARGS (appended to serve's arguments, split on spaces),
  TT_LAB_DIRTY_MARKER (default /var/lib/tenstorrent/glm-tt.dirty when that host directory is
                     mounted, else $CACHE_ROOT/glm-tt.dirty; never under /tmp). The
                     $CACHE_ROOT fallback is advisory (CI shares that cache between hosts
                     and resets the chips before each job): an old one is removed at start.
The vLLM image's arguments ``--model M --tt-device D --service-port P --no-auth`` are
accepted too, and ``glm-tt ARGS...`` / ``test-tt-kernels ARGS...`` run the binaries.

Exit status is glm-tt's: 0 after a drain (the first SIGTERM drains, a second halts); 3 when
a device group failed or the dirty marker exists: reset the chips (LoudBox tt-smi -r,
Galaxy tt-smi -glx_reset, never -r), delete the marker, then restart; 64 bad setup here.
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

DEFAULT_HOME = "/home/container_app_user/glm-tt"
HOST_MARKER_DIR = "/var/lib/tenstorrent"
SETUP_FAILED = 64

# DEVICE -> (chips, extra serve arguments, environment defaults)
DEVICES = {
    "p150x8": (8, [], {}),
    # In-group rounds over Ethernet: the Galaxy's x1 PCIe links make host rounds about
    # 1.5x slower decode.
    "blackhole_galaxy": (32, ["--groups", "4"], {"TT_LAB_GLM_ROUND": "auto"}),
}
# glm-tt's exit status when the Galaxy's link map holds no 3-cube group for serve.
NO_CUBE_GROUPS = 5
# The configuration CI's Galaxies served on before Ethernet groups: PCIe rounds and
# host-side prefill gathers, one user per group.
PCIE_FALLBACK = {"TT_LAB_GLM_ROUND": "pcie", "TT_LAB_ETH_GATHER": "0"}
DEVICES["bh_galaxy"] = DEVICES["blackhole_galaxy"]

# The fixed JWT header and payload TIS's clients sign (pyjwt's compact JSON).
JWT_HEADER = b'{"alg":"HS256","typ":"JWT"}'
JWT_PAYLOAD = b'{"team_id":"tenstorrent","token_id":"debug-test"}'


class SetupError(Exception):
    pass


def log(message):
    print(f"run_glm_tt: {message}", file=sys.stderr, flush=True)


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

    A missing or partial download then fails here, before glm-tt touches the chips (and so
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
    """GLM_TT_HOME, else GLM_TT_PACKAGE (default glm-tt) in the weights directory when present,
    else the default.

    A weights repo keeps each build in its own folder (glm-tt-<commit>) so that a cached
    download, which is complete while the old folder is there, still fetches a new build.
    """
    if env.get("GLM_TT_HOME"):
        return Path(env["GLM_TT_HOME"])
    directory = weights_dir(env)
    name = env.get("GLM_TT_PACKAGE") or "glm-tt"
    if directory and (Path(directory) / name / "bin" / "glm-tt").is_file():
        return Path(directory) / name
    return Path(DEFAULT_HOME)


def runnable_home(home: Path, staging_root=None) -> Path:
    """home if its loader and binaries can execute; else a verified, executable copy.

    A Hugging Face download keeps no exec bits and the weights mount may be read-only, so
    bin/ and runtime/ are checked against SHA256SUMS and copied to a temporary directory.
    """
    executables = [home / "runtime/ld-linux-x86-64.so.2", home / "bin/glm-tt"]
    if all(os.access(path, os.X_OK) for path in executables):
        return home
    if not (home / "SHA256SUMS").is_file():
        raise SetupError(
            f"{home} is not executable and has no SHA256SUMS to verify a copy against"
        )
    verify_package(home)
    staged = Path(tempfile.mkdtemp(prefix="glm-tt-", dir=staging_root))
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


def marker_is_shared(marker, env, explicit=False, host_marker_dir=None):
    """True for the fallback marker in CACHE_ROOT, which CI keeps on storage shared by hosts.

    Such a marker cannot say which host's chips it is about, and CI resets the chips before
    every job, so it is advisory: glm-tt's startup probe is the check that the chips work.
    """
    host_marker_dir = host_marker_dir or HOST_MARKER_DIR
    cache = env.get("CACHE_ROOT", "/home/container_app_user/cache_root")
    return (
        not explicit
        and not marker.startswith(f"{host_marker_dir}/")
        and marker == f"{cache}/glm-tt.dirty"
    )


def dirty_marker(env, host_marker_dir=None):
    """Where a halt mid-command leaves its marker: it must outlive the container."""
    host_marker_dir = host_marker_dir or HOST_MARKER_DIR
    marker = env.get("TT_LAB_DIRTY_MARKER")
    if not marker:
        cache = env.get("CACHE_ROOT", "/home/container_app_user/cache_root")
        if os.path.isdir(host_marker_dir) and os.access(host_marker_dir, os.W_OK):
            marker = f"{host_marker_dir}/glm-tt.dirty"
        elif os.path.isdir(host_marker_dir):
            raise SetupError(
                f"{host_marker_dir} is mounted but not writable by uid {os.getuid()}; make "
                f"the host directory writable (e.g. chmod 1777) or unmount it to fall back "
                f"to {cache}"
            )
        else:
            try:
                os.makedirs(cache, exist_ok=True)
            except OSError:
                pass
            if not os.access(cache, os.W_OK):
                raise SetupError(
                    f"no persistent place for the dirty-chip marker: mount a writable host "
                    f"directory at {host_marker_dir}, make {cache} writable, or set "
                    f"TT_LAB_DIRTY_MARKER (not under /tmp)"
                )
            marker = f"{cache}/glm-tt.dirty"
    if marker.startswith(("/tmp/", "/var/tmp/", "/dev/shm/")):
        raise SetupError(f"TT_LAB_DIRTY_MARKER={marker} would die with the container")
    return marker


def serve_command(env, usage, chips_present):
    """(serve arguments, environment updates) for glm-tt, from the environment.

    ``usage`` is glm-tt's usage text (its features), ``chips_present`` the number of
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
    if "--groups" in device_args and "--groups G|auto" in usage:
        device_args = ["--groups", "auto"]
    if "--groups" in device_args and "--mtp" in usage:
        # MTP's draft gathers and the batch image's rounds go over Ethernet.
        updates["TT_LAB_GLM_GATHER"] = env.get("TT_LAB_GLM_GATHER") or "eth"
    # Chunked-prefill gathers over Ethernet on every box: without it they go host-side
    # over PCIe at about 160 t/s; output is bit-identical either way.
    updates["TT_LAB_ETH_GATHER"] = env.get("TT_LAB_ETH_GATHER") or "1"

    ttq_name = env.get("GLM_TT_TTQ") or "glm53-flash-hot3-meas.ttq"
    vocab_name = env.get("GLM_TT_VOCAB") or "GLM-5.3-Flash-vocab.gguf"
    ttq, vocab = find_weight(ttq_name, env), find_weight(vocab_name, env)
    where = env.get("MODEL_WEIGHTS_DIR") or f"{env.get('HF_HOME', '<unset>')}/hub"
    if not ttq:
        raise SetupError(f"weights: {ttq_name} not found (in {where}); set GLM_TT_TTQ")
    if not vocab:
        raise SetupError(
            f"weights: {vocab_name} not found (in {where}); set GLM_TT_VOCAB"
        )

    # TIS publishes the container's port, so serve must listen beyond loopback.
    if "--host" not in usage:
        raise SetupError(
            "this glm-tt has no --host, so it listens on 127.0.0.1 only and the published "
            "port cannot reach it; use a glm-tt with --host"
        )
    args = ["--host", env.get("GLM_TT_HOST") or "0.0.0.0"]

    key, source = resolve_api_key(env)
    if key is None:
        log("NO_AUTH set, serving without an API key")
        updates["API_KEY"] = None
    else:
        log(f"API key from {source}")
        updates["API_KEY"] = key
        if "environment's API_KEY" not in usage:
            if "--api-key" not in usage:
                raise SetupError("API key set but this glm-tt has no --api-key")
            args += ["--api-key", key]

    model = env.get("MODEL") or "GLM-5.3-Flash"
    if "/" not in model:
        model = f"zai-org/{model}"
    serve = [
        "serve",
        "-m", vocab,
        "--ttq", ttq,
        "--device",
        "--tiles", "64",
        "--prefill-chunk", "2048",
        "-n", env.get("GLM_TT_MAX_TOKENS") or "163840",
        "--port", env.get("SERVICE_PORT") or "8000",
        "--served-model-name", env.get("SERVED_MODEL_NAME") or model,
    ]  # fmt: skip
    if env.get("GLM_TT_MTP"):
        mtp = find_weight(env["GLM_TT_MTP"], env)
        if not mtp:
            raise SetupError(
                f"weights: {env['GLM_TT_MTP']} not found (in {where}); set GLM_TT_MTP"
            )
        if "--mtp" not in usage:
            raise SetupError("GLM_TT_MTP is set but this glm-tt has no --mtp")
        serve += ["--mtp", mtp]
    extra = shlex.split(env.get("GLM_TT_EXTRA_ARGS", ""))
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


def _redacted(args):
    return [
        "***" if i and args[i - 1] == "--api-key" else arg for i, arg in enumerate(args)
    ]


def run_serve(command, env):
    """Run glm-tt as a child, forwarding each SIGTERM/SIGINT, and return its status."""
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

    glm-tt pins ordinary host memory for NOC DMA, which needs the IOMMU to translate (group
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


def groups_probe(glm_tt, env):
    """Print ``glm-tt groups`` (read-only: the link map, the cubes, the cover) and return
    its status: 0 with at least one cube group, 5 with none."""
    result = subprocess.run(
        glm_tt + ["groups"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in result.stdout.splitlines():
        log(f"groups: {line}")
    log(f"glm-tt groups exited {result.returncode}")
    return result.returncode


def pcie_fallback_command(serve, env):
    """(serve arguments, dropped arguments) for a Galaxy without cube groups: --groups 4,
    one user per group, no --mtp and none of GLM_TT_EXTRA_ARGS."""
    extra = shlex.split(env.get("GLM_TT_EXTRA_ARGS", ""))
    base = serve[: len(serve) - len(extra)] if extra else list(serve)
    dropped = list(extra)
    out = []
    i = 0
    while i < len(base):
        if base[i] == "--mtp":
            dropped += base[i : i + 2]
            i += 2
            continue
        if base[i] == "--groups":
            out += ["--groups", "4"]
            i += 2
            continue
        out.append(base[i])
        i += 1
    return out, dropped


def main(argv=None, environ=None):
    argv = sys.argv[1:] if argv is None else argv
    env = dict(os.environ if environ is None else environ)
    try:
        passthrough = bool(argv) and argv[0] in ("glm-tt", "test-tt-kernels")
        if not passthrough:
            parse_vllm_args(argv, env)
        home = package_home(env)
        if not (home / "bin/glm-tt").is_file():
            raise SetupError(
                f"no glm-tt in {home}/bin (put the package at GLM_TT_HOME or glm-tt/ in "
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
        glm_tt = loader + [str(home / "bin/glm-tt")]
        if verified == home:
            verify_package(home)
        if (home / "SOURCE").is_file():
            sys.stderr.write((home / "SOURCE").read_text())
        usage = subprocess.run(
            glm_tt, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        ).stdout
        chip_paths = sorted(glob.glob("/dev/tenstorrent/[0-9]*"))
        chips = len(chip_paths)
        log(device_report(chip_paths))
        if env.get("TT_LAB_LEGACY_KMD"):
            log(
                "TT_LAB_LEGACY_KMD is set but glm-tt has no legacy KMD mode and ignores it"
            )
        serve, updates = serve_command(env, usage, chips)
        ttq = serve[serve.index("--ttq") + 1] if "--ttq" in serve else ""
        if ttq.endswith(".manifest"):
            check_split_ttq(ttq)
        explicit_marker = bool(env.get("TT_LAB_DIRTY_MARKER"))
        updates["TT_LAB_DIRTY_MARKER"] = dirty_marker(env)
    except SetupError as error:
        log(str(error))
        return SETUP_FAILED
    for name, value in updates.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    log(
        f"DEVICE={env.get('DEVICE')} TT_LAB_GLM_ROUND={env.get('TT_LAB_GLM_ROUND', '')} "
        f"TT_LAB_ETH_GATHER={env['TT_LAB_ETH_GATHER']} "
        f"TT_LAB_DIRTY_MARKER={env['TT_LAB_DIRTY_MARKER']}"
    )
    marker = env["TT_LAB_DIRTY_MARKER"]
    if os.path.exists(marker) and marker_is_shared(marker, env, explicit_marker):
        log(
            f"removing {marker}: on a shared cache it may be another host's, and this "
            "host's chips are checked by glm-tt's startup probe instead"
        )
        os.remove(marker)
    elif os.path.exists(marker):
        log(
            "the dirty marker exists: a previous run halted mid-command; serve will exit 3"
        )
    galaxy = "--groups" in serve
    # A glm-tt with --groups auto also has the read-only `groups` command.
    if galaxy and "--groups G|auto" in usage:
        probe = groups_probe(glm_tt, env)
        if _truthy(env.get("GLM_TT_PROBE_ONLY")):
            return probe
    log(f"glm-tt {' '.join(_redacted(serve))}")
    status = run_serve(glm_tt + serve, env)
    fallback = (env.get("GLM_TT_FALLBACK") or "pcie").lower()
    if galaxy and status == NO_CUBE_GROUPS and fallback == "pcie":
        serve, dropped = pcie_fallback_command(serve, env)
        env.update(PCIE_FALLBACK)
        env.pop("TT_LAB_GLM_GATHER", None)
        log(
            "DOWNGRADE: glm-tt found no 3-cube groups (exit 5); serving on PCIe rounds "
            f"and host gathers, one user per group (dropped {' '.join(dropped) or 'nothing'})"
        )
        log(f"glm-tt {' '.join(_redacted(serve))}")
        status = run_serve(glm_tt + serve, env)
    if status == 3:
        log(
            "glm-tt exited 3: a device group failed or the chips are marked dirty; reset "
            "them (LoudBox tt-smi -r, Galaxy tt-smi -glx_reset) and delete "
            f"{env['TT_LAB_DIRTY_MARKER']} before restarting"
        )
    else:
        log(f"glm-tt exited {status}")
    return status


if __name__ == "__main__":
    sys.exit(main())
