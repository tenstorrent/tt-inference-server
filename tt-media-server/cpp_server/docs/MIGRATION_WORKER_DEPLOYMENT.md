# Migration Worker Deployment

This document assumes prefill and decode models and inference server are deployed and running.

## Deploy Kafka and metadata server

First, pick a host to deploy Kafka and metadata server. Common choice is the prefill rank 0.

```bash
REPO_ROOT=$PWD

cd $REPO_ROOT/tt-media-server/cpp_server/scripts

./dev-kafka.sh up

uv venv
source .venv/bin/activate
uv pip install -r migration_cli_requirements.txt

# Create topics
python migration_cli.py --brokers "$(hostname):9092" setup

# Check if they are created
python migration_cli.py --brokers "$(hostname):9092" status

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

## N-prefill exclusive ownership (#4795)

With multiple prefills, deploy pins `KAFKA_PARTITION=i` on prefill `i`, expands
request+ack topics to `>= N` partitions, and auto round-robins `WORKER_PEERS`
so each decode has a single prefill owner.

Dry-run topology preview (no containers started):

```bash
NUM_PREFILL=2 NUM_DECODE=4 ./dry_run_n_prefill_deploy.sh
```

Exabox dry-run (no model; discovery + Kafka + stub migrate):

```bash
./deploy_migration_workers.sh \
  --discovery-server "$(hostname):8080" \
  --prefill-hosts "$PREFILL_HOSTS" \
  --decode-hosts "$DECODE_HOSTS" \
  --prefill-table /data/.../prefill.pb \
  --decode-table /data/.../decode.pb \
  --kafka-brokers "$(hostname):9092" \
  --migration-mode dry-run \
  --image ghcr.io/tenstorrent/tt-shield/tt-migration-worker:<tag>
```

Local process-level ownership proof (lab + CLI produce + owner-only assert):

```bash
# Kafka must be up; PREFILL_TABLE must exist
PREFILL_TABLE=/path/to.pb \
  bash tests/e2e/scripts/run_n_prefill_ownership_e2e.sh
```

Opt-in Kafka ctests (broker required):

```bash
INTEGRATION_TESTS_ENABLED=1 KAFKA_BROKERS=localhost:9092 \
  ctest --test-dir build -L kafka --output-on-failure
```

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

