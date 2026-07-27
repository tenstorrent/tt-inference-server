# Migration Worker Deployment

This document assumes prefill and decode models and inference server are deployed and running.

## Deploy Kafka and metadata server

First, pick a host to deploy Kafka and metadata server. Common choice is the prefill rank 0.

```bash
REPO_ROOT=$PWD

cd $REPO_ROOT/tt-media-server/cpp_server/scripts

export KAFKA_ADVERTISED_HOST=$(hostname)
export KAFKA_BROKERS="${KAFKA_ADVERTISED_HOST}:9092"
./dev-kafka.sh up

uv venv
source .venv/bin/activate
uv pip install -r scripts/migration_cli_requirements.txt

# Create topics
python migration_cli.py setup

# Check if they are created
python migration_cli.py status

# Start metadata server
cd $REPO_ROOT/tt-media-server/cpp_server/scripts/metadata_server
uv pip install -r requirements.txt 
python http_metadata_server.py 
```

## Deploy migration workers

Deploying the migration workers from the prefill rank 0.

```bash
# Required to pull the internal migration worker image
# GH Token needs package:read permission
export GHCR_USERNAME=dmadictt
export GHCR_TOKEN='ghp_...'

export DECODE_HOSTS=bh-glx-110-d03u02,bh-glx-110-d03u08,bh-glx-110-d03u14,bh-glx-110-d03u20,bh-glx-110-d05u02,bh-glx-110-d05u08,bh-glx-110-d05u14,bh-glx-110-d06u02,bh-glx-110-d06u08,bh-glx-110-d06u14,bh-glx-110-d07u02,bh-glx-110-d07u08,bh-glx-110-d07u14,bh-glx-110-d08u02,bh-glx-110-d08u08,bh-glx-110-d08u14

./deploy_migration_workers.sh \
  --discovery-server "$(hostname):8080" \
  --prefill-hosts "$(hostname)" \
  --prefill-tags "$(hostname)" \
  --migration-mode device \
  --decode-hosts "$DECODE_HOSTS" \
  --decode-tags "$DECODE_HOSTS" \
  --decode-table /data/dmadic/decode-table.pb \
  --prefill-table /data/dmadic/prefill_kv_chunk_table.pb \
  --health-port 9109 \
  --prefill-device-map /tmp/prefill-device-map.txt \
  --decode-device-map /tmp/decode-device-map.txt \
  --kafka-brokers "$(hostname):9092" \
  --image ghcr.io/tenstorrent/tt-shield/tt-migration-worker:24072026-183000
```

To stop the migration workers tap `Ctrl+C`. This will also delete the remote docker containers.

# Troubleshooting

```bash
ERROR: bh-glx-110-d01u02: cannot access Docker API; add the SSH user to the docker group or configure passwordless sudo for docker
```

This happens when deploy_migration_workers.sh tries to ssh to the host it's running on. 
Just `ssh $(hostname)` then `exit` for quick fix.
Fix is on the way.

---

```bash
[deploy] loading config /data/dmadic/inf2/tt-media-server/cpp_server/scripts/migration_deploy.conf
ERROR: decode device map file not found: /tmp/device-map.txt
```

Deployment script requires the decode device map file to be present on the host it's running on. This expectation is not correct and fix is on the way in PR https://github.com/tenstorrent/tt-inference-server/pull/4767/changes.

Quick fix creating an empty file `touch /tmp/device-map.txt`.


# Building the migration worker image

```bash
cd $REPO_ROOT/tt-media-server/cpp_server/scripts
./build_migration_worker_image.sh --image tt-migration-worker:dev3 ghcr.io/tenstorrent/tt-shield/tt-migration-worker:23072026-225800

# This token requires package:write permission
docker login ghcr.io -u <GH_USERNAME>
docker push ghcr.io/tenstorrent/tt-shield/tt-migration-worker:23072026-225800
```

