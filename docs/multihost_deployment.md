# Multi-Host Deployment Guide

This document describes multi-host deployment for distributed vLLM inference on Tenstorrent hardware (Dual Galaxy, Quad Galaxy configurations).

## Overview

Multi-host deployment enables running large language models across multiple Galaxy systems connected via high-speed interconnect. The architecture uses MPI for process coordination and SSH for remote process spawning.

### Supported Configurations

| Configuration | Hosts | Chips | MESH_DEVICE | Use Case |
|---------------|-------|-------|-------------|----------|
| Dual Galaxy | 2 | 64 | `(8,8)` | Models requiring 2 Galaxy systems |
| Quad Galaxy | 4 | 128 | `(8,16)` | Large models (e.g., DeepSeek-V3) |

### Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Multi-Host Architecture                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  [User/Client]                                                             │
│       │                                                                    │
│       │ HTTP API (port 8000)                                               │
│       ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  Host 0 (Two Containers)                                           │    │
│  │  ┌───────────────────────┐       ┌───────────────────────────┐     │    │
│  │  │ Controller Container  │       │ Worker Container (rank 0) │     │    │
│  │  │ - Serves inference API│       │ - sshd (port 2200)        │     │    │
│  │  │ - Runs mpirun to spawn│──────▶│ - Runs MPI rank 0 process │     │    │
│  │  │   MPI processes       │       │ - TT Device access        │     │    │
│  │  └───────────┬───────────┘       └───────────────────────────┘     │    │
│  └──────────────┼─────────────────────────────────────────────────────┘    │
│                 │                                                          │
│                 │ SSH (port 2200) to spawn MPI processes                   │
│                 ▼                                                          │
│  ┌────────────────────────────────┐  ┌────────────────────────────────┐    │
│  │  Host 1: Worker (rank 1)       │  │  Host 2, 3: Workers            │    │
│  │  ┌──────────────────────────┐  │  │  (Quad Galaxy only)            │    │
│  │  │ Worker Container         │  │  │  ┌──────────────────────────┐  │    │
│  │  │ - sshd (port 2200)       │  │  │  │ Same as Host 1           │  │    │
│  │  │ - Runs MPI rank 1 process│  │  │  │ - rank 2, rank 3         │  │    │
│  │  │ - TT Device access       │  │  │  └──────────────────────────┘  │    │
│  │  └──────────────────────────┘  │  │                                │    │
│  └────────────────────────────────┘  └────────────────────────────────┘    │
│                                                                            │
│  Note: All hosts run Worker containers. Controller spawns MPI processes    │
│        on all Workers (including rank 0 on Host 0) via SSH (port 2200).    │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Host 0 runs **two containers**: Controller (vLLM API server, mpirun) + Worker (sshd, MPI rank 0)
- Hosts 1+ run Worker containers (sshd, MPI rank 1+)
- Controller runs `mpirun` which spawns MPI processes on all Workers via SSH
- All Workers (including rank-0) run the actual MPI processes for distributed inference
- **Important:** `run.py` must be executed on the rank-0 host (first in `MULTIHOST_HOSTS`)

---

## Prerequisites (User-Provided)

### Infrastructure Requirements

| Requirement | Description |
|-------------|-------------|
| **Hosts** | 2 or 4 Galaxy systems with network connectivity |
| **TT Devices** | `/dev/tenstorrent` available on all hosts |
| **Hugepages** | `/dev/hugepages-1G` configured and mounted on all hosts |
| **Docker** | Docker installed and running on all hosts |
| **Docker Image** | Multi-host image built from `vllm.tt-metal.src.multihost.Dockerfile` |
| **SSH Access** | Orchestrator host can SSH to all target hosts (for container management) |
| **Shared Storage** | NFS or similar shared filesystem mounted on all hosts |

### Network Requirements

| Port | Protocol | Purpose |
|------|----------|---------|
| 2200 | TCP | SSH for MPI process spawning (Worker containers) |
| 8000 | TCP | vLLM API server (Controller container) |
| MPI ports | TCP | MPI communication (dynamic, via `MPI_INTERFACE`) |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Yes | HuggingFace token for gated model downloads |
| `JWT_SECRET` | Conditional | Required unless `--no-auth` is specified |

---

## Configuration (Environment Variables)

Multi-host configuration is provided via environment variables in the `.env` file. If a required variable is not set, the orchestrator will prompt for interactive input.

### Setting Up `.env` File

```bash
# Required multi-host configuration
MULTIHOST_HOSTS=host1,host2           # Comma-separated list of hostnames
MPI_INTERFACE=cnx1                     # Network interface for MPI communication
SHARED_STORAGE_ROOT=/mnt/shared        # Path to shared storage (must exist on all hosts)

# Optional
CONFIG_PKL_DIR=/mnt/shared/config_pkl  # Auto-generated under SHARED_STORAGE_ROOT if not set
TT_SMI_PATH=tt-smi                     # Path to tt-smi binary on hosts (default: "tt-smi")

# DeepSeek models only
DEEPSEEK_V3_HF_MODEL=/mnt/shared/models/deepseek-v3
DEEPSEEK_V3_CACHE=/mnt/shared/cache/deepseek-v3
```

### Running Multi-Host Deployment

```bash
python3 run.py \
    --model <MODEL_NAME> \
    --workflow server \
    --docker-server \
    --tt-device <dual_galaxy|quad_galaxy>
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MULTIHOST_HOSTS` | Comma-separated list of hostnames | `host1,host2` |
| `MPI_INTERFACE` | Network interface for MPI communication | `cnx1`, `eth0` |
| `SHARED_STORAGE_ROOT` | Path to shared storage (must exist on all hosts) | `/mnt/shared` |

### Optional Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CONFIG_PKL_DIR` | Directory for vLLM config pickle files | Auto-generated under `SHARED_STORAGE_ROOT` |
| `TT_SMI_PATH` | Path to tt-smi binary on remote hosts | `tt-smi` |
| `DEEPSEEK_V3_HF_MODEL` | Path to DeepSeek-V3 HuggingFace model (DeepSeek only) | - |
| `DEEPSEEK_V3_CACHE` | Path to DeepSeek-V3 cache directory (DeepSeek only) | - |

### Validation Rules

- **`run.py` must be executed on the rank-0 host** (first hostname in `MULTIHOST_HOSTS`). This is required because the Controller container runs locally and MPI rank-0 must be on the same host for torch distributed TCP rendezvous to work correctly.
- `MULTIHOST_HOSTS` must contain exactly 2 or 4 hostnames (matching `--tt-device`)
- `SHARED_STORAGE_ROOT` must exist and be accessible on all hosts
- `CONFIG_PKL_DIR` must be under `SHARED_STORAGE_ROOT`
- All hosts must be reachable via SSH from the orchestrator
- DeepSeek paths must be under `SHARED_STORAGE_ROOT`

---

## Automatic Configuration (tt-inference-server)

The following configurations are automatically generated by `MultiHostOrchestrator.prepare()`:

### 1. SSH Key Pair (Ephemeral)

**Purpose:** Secure communication between Controller and Worker containers

| Item | Details |
|------|---------|
| Generation | `ssh-keygen -t ed25519` at runtime |
| Location | `/tmp/tt_multihost_XXXXX/id_ed25519_multihost{,.pub}` |
| Lifecycle | Created on `prepare()`, deleted on session cleanup |
| Permissions | Private key: `0600` |

**Integration Note:** For Kubernetes deployments, you may pre-generate SSH keys as Secrets and mount them into pods. The Worker entrypoint expects the public key at `/tmp/authorized_keys.pub`.

### 2. SSH Configuration

**Purpose:** Simplify SSH connections from Controller to Workers

**Generated content:**
```
Host host1
    Port 2200
    User container_app_user
    IdentityFile /home/container_app_user/.ssh/id_ed25519_multihost
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    BatchMode yes

Host host2
    Port 2200
    User container_app_user
    IdentityFile /home/container_app_user/.ssh/id_ed25519_multihost
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    BatchMode yes
```

**Note:** Uses real hostnames directly (not aliases) so that MPI rankfile resolution works without requiring `/etc/hosts` entries.

**Mount location in Controller:** `/home/container_app_user/.ssh/config`

### 3. MPI Rankfile

**Purpose:** Map MPI ranks to hosts

**Generated content:**
```
# mpirun rankfile
rank 0=host1 slot=0:*
rank 1=host2 slot=0:*
rank 2=host3 slot=0:*  # Quad Galaxy only
rank 3=host4 slot=0:*  # Quad Galaxy only
```

**Note:** Uses real hostnames directly (matching `MULTIHOST_HOSTS`).

**Mount location in Controller:** `/etc/mpirun/rankfile`

### 4. override_tt_config

**Purpose:** Configure vLLM TT backend for multi-host operation

**Generated JSON:**
```json
{
  "rank_binding": "/home/container_app_user/tt-metal/tests/tt_metal/distributed/config/<config>.yaml",
  "mpi_args": "--host host1,host2 --map-by rankfile:file=/etc/mpirun/rankfile --bind-to none --tag-output",
  "extra_ttrun_args": "--tcp-interface ${MPI_INTERFACE}",
  "config_pkl_dir": "${CONFIG_PKL_DIR}",
  "fabric_config": "FABRIC_1D",
  "fabric_reliability_mode": "RELAXED_INIT",
  "env_passthrough": ["VLLM_*", "MESH_DEVICE", "HF_TOKEN", ...],
  "trace_mode": "none"
}
```

**Note:** `mpi_args` uses real hostnames from `MULTIHOST_HOSTS`.

### 5. Automatic Value Determination

| Value | Logic | Result |
|-------|-------|--------|
| `MESH_DEVICE` | Based on host count | 2 hosts → `(8,4)`, 4 hosts → `(8,8)` |
| `rank_binding` | Based on host count | `dual_galaxy_rank_bindings.yaml` or `quad_galaxy_rank_bindings.yaml` |
| SSH port | Constant | `2200` |
| Container user | Constant | `container_app_user` |

---

## Building the Multi-Host Docker Image

The multi-host deployment requires a Docker image with sshd and the worker entrypoint script.

```bash
# Build multi-host image from base vLLM image
docker build -t <your-registry>/vllm-multihost:<tag> \
    --build-arg BASE_IMAGE=ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:<version> \
    -f vllm-tt-metal/vllm.tt-metal.src.multihost.Dockerfile .
```

The Dockerfile adds:
- OpenSSH server (`sshd`)
- SSH configuration (port 2200, pubkey auth only)
- Unified entrypoint script (`/usr/local/bin/multihost_entrypoint.sh`)
- MPI config directory (`/etc/mpirun`)

---

## Container Specifications

### Worker Container

**Purpose:** Run sshd to accept MPI process spawning from Controller

**Startup command:**
```bash
docker run --rm -d \
    --name tt-worker-<rank> \
    --user root \
    --net host \
    --pid host \
    --device /dev/tenstorrent:/dev/tenstorrent \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount type=bind,src=${CONFIG_PKL_DIR},dst=${CONFIG_PKL_DIR} \
    --mount type=bind,src=<pubkey-path>,dst=/tmp/authorized_keys.pub,readonly \
    -e MULTIHOST_ROLE=worker \
    -e SSH_PORT=2200 \
    --entrypoint /usr/local/bin/multihost_entrypoint.sh \
    <docker-image>
```

**Unified Entrypoint (`multihost_entrypoint.sh`):**

The entrypoint script handles both Worker and Controller roles via the `MULTIHOST_ROLE` environment variable:
- **Worker mode**: Copies public key to authorized_keys, starts sshd
- **Controller mode**: Copies SSH config, drops privileges, executes vLLM command

**Key points:**
- Runs as `root` (required for sshd)
- Uses `--net host` for MPI communication using a Host Networking
- Public key mounted read-only, copied to correct location with proper permissions

### Controller Container

**Purpose:** Run vLLM API server and coordinate MPI processes

**Startup command:**
```bash
docker run --rm \
    --name tt-controller \
    --net host \
    --pid host \
    --ipc host \
    --env-file .env \
    --device /dev/tenstorrent:/dev/tenstorrent \
    --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
    --mount type=bind,src=<config-dir>,dst=/home/container_app_user/.ssh,readonly \
    --mount type=bind,src=<config-dir>/mpirun,dst=/etc/mpirun,readonly \
    --mount type=bind,src=${CONFIG_PKL_DIR},dst=${CONFIG_PKL_DIR} \
    -e MULTIHOST_ROLE=controller \
    -e MESH_DEVICE=<mesh-device> \
    -e VLLM_TARGET_DEVICE=tt \
    -e NCCL_SOCKET_IFNAME=${MPI_INTERFACE} \
    -e ETH=${MPI_INTERFACE} \
    -e HOSTS=${MULTIHOST_HOSTS} \
    -e CONFIG_JSON='<override_tt_config_json>' \
    <docker-image> \
    --model <model> \
    --tt-device <device>
```

**Key points:**
- `--rm`: Container removed on exit
- `--net host`: Uses host network, so real hostnames resolve via host DNS
- `--ipc host`: Shared memory for inter-process communication
- `--env-file .env`: Load `HF_TOKEN`, `JWT_SECRET` from environment file
- SSH config directory mounted for Worker connectivity
- `CONFIG_JSON` environment variable contains the override_tt_config JSON
- `NCCL_SOCKET_IFNAME` and `ETH`: Network interface for MPI/NCCL communication

---

## Configuration Reference

### Constants

| Constant | Value | Location |
|----------|-------|----------|
| `WORKER_SSH_PORT` | `2200` | `workflows/multihost_config.py` |
| `CONTAINER_USER` | `container_app_user` | `workflows/multihost_config.py` |
| `DEFAULT_IDENTITY_FILE` | `/home/container_app_user/.ssh/id_ed25519_multihost` | `workflows/multihost_config.py` |

### Environment Variables Passed to MPI Workers

```python
ENV_PASSTHROUGH = [
    "VLLM_*",
    "MESH_DEVICE",
    "HF_TOKEN",
    "DEEPSEEK_V3_CACHE",
    "DEEPSEEK_V3_HF_MODEL",
    "TT_METAL_HOME",
    "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME",
]
```

### Rank Binding Files

| Hosts | File |
|-------|------|
| 2 (Dual Galaxy) | `tt-metal/tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml` |
| 4 (Quad Galaxy) | `tt-metal/tests/tt_metal/distributed/config/quad_galaxy_rank_bindings.yaml` |

---

## Troubleshooting

### Controller Host Mismatch (run.py on Wrong Host)

If you see an error like:
```
run.py must be executed on the rank-0 host.
  Current host: host2
  Rank-0 host (MULTIHOST_HOSTS[0]): host1
```

This means `run.py` is being executed on a host that doesn't match the first hostname in `MULTIHOST_HOSTS`. The Controller container runs locally and must be on the same host as MPI rank-0 for torch distributed to work correctly.

**Solutions:**
1. SSH to the rank-0 host and run `run.py` from there:
   ```bash
   ssh host1
   cd /path/to/tt-inference-server
   python3 run.py --model <MODEL> --workflow server --docker-server --tt-device dual_galaxy
   ```

2. Or update `MULTIHOST_HOSTS` to start with the current host:
   ```bash
   # If you're on host2 and want it to be rank-0
   export MULTIHOST_HOSTS=host2,host1
   ```

### TT Device Initialization Error

If you encounter device initialization errors (e.g., "device not available", "failed to acquire device"), it may be due to incomplete cleanup from a previous run. To resolve:

1. Reset TT devices on all Worker hosts:
   ```bash
   # Run on each worker host
   tt-smi -glx_reset
   ```

2. Wait for reset to complete (~30 seconds per host)

3. Retry the deployment

### SSH Connection Failed

1. Verify Worker container is running: `docker ps | grep tt-worker`
2. Check sshd is listening: `docker exec tt-worker-1 ss -tlnp | grep 2200`
3. Verify public key was copied: `docker exec tt-worker-1 cat /home/container_app_user/.ssh/authorized_keys`
4. Test SSH manually: `ssh -i <key> -p 2200 container_app_user@<host> hostname`

### MPI Spawn Failed

1. Check Controller can reach Workers: `docker exec tt-controller ssh <host2> hostname`
2. Verify rankfile contents: `docker exec tt-controller cat /etc/mpirun/rankfile`
3. Check MPI interface is correct: `ip addr show ${MPI_INTERFACE}`

### Config Pickle Directory Issues

1. Verify mount on all hosts: `ls -la ${CONFIG_PKL_DIR}`
2. Check permissions: directory must be writable by vLLM process (UID 1000)
