# Migration Worker Integration Tests

End-to-end tests for the Kafka + Mooncake migration pipeline.  
Tests exercise: **Kafka message delivery → MigrationWorker → MooncakeTransferEngine RDMA pull**.

---

## Prerequisites

| Dependency | How to get it |
|---|---|
| Docker | Already installed in dev containers |
| Kafka | Docker container (instructions below) |
| Build flags | `./build.sh --blaze --kafka --mooncake` |

---

## 1. Start Kafka

We run Apache Kafka 3.7.0 in KRaft mode (no Zookeeper) with `--network host` so it's reachable from sibling containers at the Docker gateway IP.

```bash
# Remove any stale container
docker rm -f kafka 2>/dev/null || true

# Start Kafka (KRaft mode, single node)
docker run -d --name kafka --network host \
  -e KAFKA_NODE_ID=1 \
  -e KAFKA_PROCESS_ROLES=broker,controller \
  -e KAFKA_LISTENERS=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093 \
  -e KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://172.17.0.1:9092 \
  -e KAFKA_CONTROLLER_LISTENER_NAMES=CONTROLLER \
  -e KAFKA_LISTENER_SECURITY_PROTOCOL_MAP=CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT \
  -e KAFKA_CONTROLLER_QUORUM_VOTERS=1@localhost:9093 \
  -e KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR=1 \
  -e KAFKA_LOG_DIRS=/tmp/kraft-combined-logs \
  -e CLUSTER_ID=MkU3OEVBNTcwNTJENDM2Qk \
  apache/kafka:3.7.0
```

**Wait ~5s** for Kafka to be ready, then verify:

```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 --list
```

### Important: `advertised.listeners`

- From **inside a dev container**: Kafka is reachable at `172.17.0.1:9092` (Docker gateway)
- From the **host machine**: use `localhost:9092`
- Set `KAFKA_BROKERS` accordingly when running tests

---

## 2. Create the Topic

The default topic is `session-offload` with 3 partitions (for distributing across consumers):

```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic session-offload \
  --partitions 3 --replication-factor 1
```

Verify:

```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --describe --topic session-offload
```

---

## 3. Build

```bash
cd cpp_server
./build.sh --blaze --kafka --mooncake
```

This produces:
- `build/tt_consumer` — Kafka consumer with MooncakeTransferEngine
- `build/migration_producer_dummy` — Native C++ Kafka producer (configurable source)
- `build/migration_source_dummy` — Mooncake segment source (registers memory for pull tests)

---

## 4. Run the E2E Test

### Mooncake Migration E2E (recommended)

Tests the full pipeline: source registers memory → producer publishes to Kafka → 3 consumers pull via Mooncake TCP.

```bash
cd cpp_server
KAFKA_BROKERS=172.17.0.1:9092 ./tests/integration/run_mooncake_migration_e2e.sh ./build
```

Expected output:
```
RESULT: PASS ✅ (all 9 transfers succeeded across 3 consumers)
```

### Kafka-only Distribution Test

Tests Kafka message distribution across consumers (no Mooncake, no real transfers):

```bash
cd cpp_server
KAFKA_BROKERS=172.17.0.1:9092 \
KAFKA_OFFLOAD_TOPIC_NAME=offload_requests \
  ./tests/integration/run_migration_workers.sh ./build
```

---

## 5. Running on Another Machine

### Same-machine (loopback) — what the E2E test does

Both source and consumers run in the same container, using `127.0.0.1`. This is the default.

### Cross-machine (real network)

For testing between two separate hosts (e.g., prefill node → decode node):

**On Machine A (source / prefill node):**

```bash
# Start the source with a routable IP
./build/migration_source_dummy \
  --local-server-name "10.0.0.5:0" \
  --buffer-size 52428800 \
  --fill-byte 171

# Output: SEGMENT=10.0.0.5:16234
```

**On Machine B (consumer / decode node):**

```bash
# Start the consumer pointing to Kafka
KAFKA_BROKERS=<kafka-host>:9092 \
./build/tt_consumer \
  --local-server-name "10.0.0.6:0" \
  -p 8001
```

**Publish a migration request (from anywhere with Kafka access):**

```bash
./build/migration_producer_dummy \
  --brokers <kafka-host>:9092 \
  --topic session-offload \
  --count 1 \
  --interval-ms 0 \
  --source-host 10.0.0.5 \
  --source-port 16234 \
  --source-offset 0 \
  --kv-size 52428800
```

The consumer on Machine B will connect to the source on Machine A via Mooncake P2P handshake and pull 50MB over TCP (or RDMA if built with `--mooncake-rdma` and NIC is available).

---

## 6. Environment Variables

| Variable | Default | Description |
|---|---|---|
| `KAFKA_BROKERS` | `localhost:9092` | Kafka bootstrap server(s) |
| `KAFKA_OFFLOAD_TOPIC_NAME` | `session-offload` | Topic for migration messages |
| `KAFKA_GROUP_ID` | `migration-workers` | Consumer group ID |
| `TT_LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warn`, `error`) |
| `NUM_CONSUMERS` | `3` | Number of consumer instances (E2E test) |
| `NUM_MESSAGES` | `9` | Number of messages to publish (E2E test) |

---

## 7. Troubleshooting

### "Connection refused" to Kafka

- Inside a dev container, use `KAFKA_BROKERS=172.17.0.1:9092`
- Ensure Kafka's `advertised.listeners` matches the reachable IP
- Check: `docker exec kafka /opt/kafka/bin/kafka-topics.sh --bootstrap-server localhost:9092 --list`

### "Unknown topic or partition"

- Create the topic first (Step 2)
- Or wait a few seconds — the consumer will retry

### "Failed to open segment"

- Ensure the source process is still running
- Check that the `source_host:source_port` in the Kafka message matches the source's actual `SEGMENT=` output
- For cross-machine: verify network connectivity between the two hosts on the Mooncake ports

### Stale messages from previous runs

The E2E test script purges the topic automatically. For manual runs:

```bash
docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --delete --topic session-offload

docker exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --topic session-offload \
  --partitions 3 --replication-factor 1
```

---

## 8. Performance Reference (TCP loopback)

| Transfer size | Time (pre-alloc) | Throughput |
|---|---|---|
| 64 KB | < 1 ms | — |
| 50 MB | ~48 ms | ~1.04 GB/s |
| 500 MB | ~460 ms | ~1.09 GB/s |

With RDMA over 100Gbps NIC, expect ~40ms for 500MB.
