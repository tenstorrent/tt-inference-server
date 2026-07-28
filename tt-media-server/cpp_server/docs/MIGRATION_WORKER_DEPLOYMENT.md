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
uv pip install -r migration_cli_requirements.txt

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

Attention: Prefill and decode table paths must be accessible by all the migration workers. For exabox deployments, this requires storing them on the /data partition.

# Building the migration worker image

```bash
cd $REPO_ROOT/tt-media-server/cpp_server/scripts

# Just build the image
./build_migration_worker_image.sh

# Or build and push the image
# This token requires package:write permission
docker login ghcr.io -u <GH_USERNAME>
./build_migration_worker_image.sh --push
```

