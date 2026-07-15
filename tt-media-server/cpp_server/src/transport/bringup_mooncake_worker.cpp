// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// bringup_mooncake_worker: production entry point / composition root for a
// migration worker process. Two responsibilities — *only* at this binary
// layer:
//
//   1. Mooncake bring-up. Parse + validate CLI, build the transfer engine and
//      PeerDiscoveryService, hand them to a MooncakeMigrationWorker that owns
//      the phased lifecycle (allocate host-DRAM pool → init engine →
//      register/publish → discover peers). Worker dtor handles teardown in
//      reverse on process exit.
//
//   2. KV-migration request loop. Once Mooncake is READY, the main thread
//      drains MigrationRequestMessages from Kafka, logs them, and publishes
//      acks — inlined here on purpose. The shape mirrors KvMigrationWorker's
//      consumerLoop in src/runtime/worker/kv_migration_worker.cpp (used by
//      tt_kv_migration_consumer), but the two binaries deliberately don't
//      share that code: this loop will grow a dispatch through `worker`
//      (writeTensorOnSender / transferToReceiver / verifyTensorOnReceiver)
//      while the consumer binary stays generic via its IMigrationExecutor
//      injection point. Keeping the loop here also means
//      MooncakeMigrationWorker has zero Kafka surface area — the only file
//      that knows about Kafka in this binary is this one.
//
//      Workers that should not consume requests (e.g. decode-side peers in
//      a prefill→decode topology) pass --no-kafka. They skip Kafka client
//      construction entirely and just hold open their Mooncake segment
//      until SIGTERM via a heartbeat-only idle loop.
//
// The main thread is the consumer thread; it polls Kafka with a short
// timeout that also paces the periodic "still alive" heartbeat. On
// SIGTERM/SIGINT the stop flag flips, the loop exits, and stack-scope
// destructors tear down in reverse order (Kafka clients, then
// MooncakeMigrationWorker unregisters its segment).

#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "config/defaults.hpp"
#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "messaging/migration_message.hpp"
#include "services/remote_kv_manager.hpp"
#include "transport/host_dram_storage_backend.hpp"
#include "transport/mooncake_migration_worker.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/peer_discovery_service.hpp"
#include "transport/transfer_types.hpp"
#include "transport/worker_health_server.hpp"
#include "utils/logger.hpp"

namespace {

using namespace tt::transport;

constexpr std::size_t K_DEFAULT_HOST_DRAM_BYTES = 4ULL << 30;  // 4 GiB
constexpr std::size_t K_RAM_HEADROOM_BYTES = 1ULL << 30;       // 1 GiB
constexpr int K_DEFAULT_DISCOVERY_TIMEOUT_SEC = 30;
constexpr int K_DISCOVERY_POLL_INTERVAL_MS = 1000;
// Kafka poll timeout also paces the heartbeat granularity: receive() blocks for
// up to K_KAFKA_POLL_MS, which is short enough that gStopRequested is honored
// promptly and the heartbeat fires within K_KAFKA_POLL_MS of K_HEARTBEAT_SEC.
constexpr int K_KAFKA_POLL_MS = 100;
constexpr int K_HEARTBEAT_SEC = 30;

// Mirror of tt::config::*() lookups without dragging in llm_runner_lib for one
// process-orchestration binary: same KAFKA_* env-var contract, same defaults
// from include/config/defaults.hpp.
std::string envOr(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v != nullptr && *v != '\0') ? std::string{v} : std::string{fallback};
}

struct WorkerConfig {
  std::string metadata_uri;        ///< Discovery service (REQUIRED).
  std::string name;                ///< This worker's logical segment name.
  std::vector<std::string> peers;  ///< Peers to discover on bring-up.
  std::size_t host_dram_bytes = K_DEFAULT_HOST_DRAM_BYTES;
  int discovery_timeout_sec = K_DEFAULT_DISCOVERY_TIMEOUT_SEC;
  TransportProtocol protocol = TransportProtocol::TCP;
  // KV layer span [layer_start, layer_end); 0 == unset. uint32_t matches
  // MigrationRequestMessage's layer_begin/layer_end so no truncation when
  // mapped to config.
  uint32_t layer_start = 0;
  uint32_t layer_end = 0;
  // When false (--no-kafka), the worker brings Mooncake up and then idles
  // until SIGTERM without ever creating Kafka clients. Used for receiver
  // roles in a prefill→decode topology.
  bool kafka_enabled = true;
  // HTTP health surface (/healthz, /readyz, /metrics). 0 == disabled, so local
  // runs and the discovery e2e tests stay port-free unless a port is asked for.
  std::uint16_t health_port = 0;
  std::string health_host = "0.0.0.0";
};

std::atomic<bool> gStopRequested{false};

// Touched from a signal handler, so it must be lock-free: [support.signal]
// permits a handler to access a lock-free atomic (not just sig_atomic_t).
static_assert(std::atomic<bool>::is_always_lock_free,
              "gStopRequested must be lock-free to be signal-safe");

void onSignal(int /*signum*/) {
  gStopRequested.store(true, std::memory_order_relaxed);
}

void usage() {
  std::cerr
      << "usage: bringup_mooncake_worker\n"
         "  --metadata URI         discovery service (REQUIRED), e.g.\n"
         "                         http://META_HOST:8080/metadata\n"
         "  --name NAME            this worker's logical segment name "
         "(REQUIRED)\n"
         "  --peer NAME            peer to discover; repeatable (REQUIRED)\n"
         "  [--host-dram-bytes N]  pool size, page-aligned (default 4 GiB)\n"
         "  [--protocol tcp|rdma]  transport (default tcp)\n"
         "  [--discovery-timeout-sec S] (default 30)\n"
         "  [--layer-start N]      first KV layer this worker owns (default "
         "0)\n"
         "  [--layer-end M]        one past last KV layer (exclusive; "
         "0=unset)\n"
         "  [--no-kafka]           skip Kafka clients; idle after bring-up\n"
         "  [--health-port N]      serve /healthz /readyz /metrics on N "
         "(0=off, default off)\n"
         "  [--health-host HOST]   bind address for the health server "
         "(default 0.0.0.0)\n"
         "  [-h|--help]            show this help and exit\n"
         "\n"
         "Multi-NIC hosts: set MC_TCP_BIND_ADDRESS to the IP peers connect "
         "to.\n";
}

// Page-alignment and RAM-headroom checks keep a misconfigured pool from
// silently over-committing the host (#4294 review note).
bool validateHostDramBytes(std::size_t bytes, std::string& err) {
  if (bytes == 0) {
    err = "must be > 0";
    return false;
  }
  const long pageSize = ::sysconf(_SC_PAGESIZE);
  if (pageSize > 0 && bytes % static_cast<std::size_t>(pageSize) != 0) {
    err = "must be page-aligned to " + std::to_string(pageSize);
    return false;
  }
  const long physPages = ::sysconf(_SC_PHYS_PAGES);
  if (pageSize > 0 && physPages > 0) {
    const auto totalRam = static_cast<std::size_t>(physPages) *
                          static_cast<std::size_t>(pageSize);
    if (bytes + K_RAM_HEADROOM_BYTES > totalRam) {
      err = "exceeds physical RAM minus 1 GiB headroom";
      return false;
    }
  }
  return true;
}

bool parseProtocol(const std::string& value, TransportProtocol& out) {
  if (value == "tcp") {
    out = TransportProtocol::TCP;
    return true;
  }
  if (value == "rdma") {
    out = TransportProtocol::RDMA;
    return true;
  }
  return false;
}

// Strict unsigned parse: the WHOLE token must be a plain non-negative integer.
// strtoull alone would silently accept "4Gib" as 4 (it stops at the first
// non-digit); we reject any leftover so a typo fails loud instead of allocating
// 4 bytes (#4294 bug).
bool parseSizeBytes(const std::string& value, std::size_t& out,
                    std::string& err) {
  if (value.empty() || value.front() == '-') {
    err = "must be a non-negative integer";
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end, 0);
  if (errno == ERANGE) {
    err = "out of range";
    return false;
  }
  if (end == value.c_str() || *end != '\0') {
    err = "must be a plain integer (no suffixes like 'GiB')";
    return false;
  }
  out = static_cast<std::size_t>(parsed);
  return true;
}

// Strict uint32 parse: whole token must fit in uint32_t, else fail loud
// instead of silently wrapping when stored as a layer id.
bool parseUint32(const std::string& value, uint32_t& out, std::string& err) {
  if (value.empty() || value.front() == '-') {
    err = "must be a non-negative integer";
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end, 10);
  if (errno == ERANGE || parsed > std::numeric_limits<uint32_t>::max()) {
    err = "must fit in a 32-bit unsigned integer";
    return false;
  }
  if (end == value.c_str() || *end != '\0') {
    err = "must be a plain integer";
    return false;
  }
  out = static_cast<uint32_t>(parsed);
  return true;
}

// Strict positive-int parse: the WHOLE token must parse and be > 0. atoi would
// turn "abc" (and an explicit "0") into 0, which means "give up discovery
// immediately" — so an invalid value silently broke bring-up (#4294 bug).
bool parsePositiveInt(const std::string& value, int& out, std::string& err) {
  if (value.empty()) {
    err = "must be a positive integer";
    return false;
  }
  errno = 0;
  char* end = nullptr;
  const long parsed = std::strtol(value.c_str(), &end, 10);
  if (errno == ERANGE || parsed > std::numeric_limits<int>::max()) {
    err = "out of range";
    return false;
  }
  if (end == value.c_str() || *end != '\0') {
    err = "must be a plain integer";
    return false;
  }
  if (parsed <= 0) {
    err = "must be > 0";
    return false;
  }
  out = static_cast<int>(parsed);
  return true;
}

// Phase 1: parse and validate everything before any resource is touched.
bool parseConfig(int argc, char** argv, WorkerConfig& cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    if (a == "-h" || a == "--help") {
      usage();
      std::exit(0);
    }
    std::string v;
    if (a == "--metadata" && next(cfg.metadata_uri)) continue;
    if (a == "--name" && next(cfg.name)) continue;
    if (a == "--peer" && next(v)) {
      cfg.peers.push_back(v);
      continue;
    }
    if (a == "--host-dram-bytes" && next(v)) {
      std::string perr;
      if (!parseSizeBytes(v, cfg.host_dram_bytes, perr)) {
        std::cerr << "--host-dram-bytes invalid ('" << v << "'): " << perr
                  << "\n";
        return false;
      }
      continue;
    }
    if (a == "--protocol" && next(v) && parseProtocol(v, cfg.protocol))
      continue;
    if (a == "--discovery-timeout-sec" && next(v)) {
      std::string perr;
      if (!parsePositiveInt(v, cfg.discovery_timeout_sec, perr)) {
        std::cerr << "--discovery-timeout-sec invalid ('" << v << "'): " << perr
                  << "\n";
        return false;
      }
      continue;
    }
    if (a == "--layer-start" && next(v)) {
      std::string perr;
      if (!parseUint32(v, cfg.layer_start, perr)) {
        std::cerr << "--layer-start invalid ('" << v << "'): " << perr << "\n";
        return false;
      }
      continue;
    }
    if (a == "--layer-end" && next(v)) {
      std::string perr;
      if (!parseUint32(v, cfg.layer_end, perr)) {
        std::cerr << "--layer-end invalid ('" << v << "'): " << perr << "\n";
        return false;
      }
      continue;
    }
    if (a == "--no-kafka") {
      cfg.kafka_enabled = false;
      continue;
    }
    if (a == "--health-port" && next(v)) {
      uint32_t port = 0;
      std::string perr;
      if (!parseUint32(v, port, perr) ||
          port > std::numeric_limits<std::uint16_t>::max()) {
        std::cerr << "--health-port invalid ('" << v
                  << "'): must be 0..65535\n";
        return false;
      }
      cfg.health_port = static_cast<std::uint16_t>(port);
      continue;
    }
    if (a == "--health-host" && next(cfg.health_host)) continue;
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }
  if (cfg.metadata_uri.empty() || cfg.name.empty() || cfg.peers.empty()) {
    std::cerr << "--metadata, --name and at least one --peer are required\n";
    return false;
  }
  if (cfg.layer_end != 0 && cfg.layer_end <= cfg.layer_start) {
    std::cerr << "--layer-end (" << cfg.layer_end
              << ") must be greater than --layer-start (" << cfg.layer_start
              << ")\n";
    return false;
  }
  std::string err;
  if (!validateHostDramBytes(cfg.host_dram_bytes, err)) {
    std::cerr << "--host-dram-bytes invalid: " << err << "\n";
    return false;
  }
  return true;
}

// Translate parsed CLI flags into the worker's domain config. The worker owns
// the lifecycle (allocate/init/register/discover/run/teardown); main only wires
// dependencies and maps config — it does not sequence the phases. The discovery
// timeout is not part of this config: it belongs to the PeerDiscoveryService.
MigrationWorkerConfig toWorkerConfig(const WorkerConfig& cli) {
  MigrationWorkerConfig cfg;
  cfg.metadata_uri = cli.metadata_uri;
  cfg.segment_name = cli.name;
  cfg.protocol = cli.protocol;
  cfg.host_dram_bytes = cli.host_dram_bytes;
  cfg.peer_segment_names = cli.peers;
  cfg.layer_start = cli.layer_start;
  cfg.layer_end = cli.layer_end;
  return cfg;
}

void installSignalHandlers() {
  std::signal(SIGTERM, onSignal);
  std::signal(SIGINT, onSignal);
}

struct KafkaConfig {
  std::string brokers;
  std::string requestTopic;
  std::string ackTopic;
  std::string groupId;
};

// Same env-var contract as tt::config::* in src/config/settings.cpp; mirrored
// here so this binary doesn't need to link llm_runner_lib just for four
// lookups.
KafkaConfig loadKafkaConfig() {
  return KafkaConfig{
      .brokers = envOr("KAFKA_BROKERS", tt::config::defaults::KAFKA_BROKERS),
      .requestTopic =
          envOr("KAFKA_MIGRATION_REQUEST_TOPIC",
                tt::config::defaults::KAFKA_MIGRATION_REQUEST_TOPIC),
      .ackTopic = envOr("KAFKA_MIGRATION_ACK_TOPIC",
                        tt::config::defaults::KAFKA_MIGRATION_ACK_TOPIC),
      .groupId = envOr("KAFKA_GROUP_ID", tt::config::defaults::KAFKA_GROUP_ID),
  };
}

// Parse one migration request, drop it unless this worker owns the request's
// layer, then publish a SUCCESSFUL ack. `worker` is also the future dispatch
// point for the data plane (writeTensorOnSender / transferToReceiver /
// verifyTensorOnReceiver) once it is wired.
void handleMigrationRequest(const std::string& raw,
                            MooncakeMigrationWorker& worker,
                            tt::messaging::IKafkaProducer& ackProducer) {
  const auto parsed = tt::messaging::parseMigrationRequest(raw);
  if (!parsed.has_value()) {
    TT_LOG_WARN("[bringup] dropping unparseable request: {}", raw);
    return;
  }

  // A single request is broadcast to every worker in the role (one consumer
  // group each); only the worker owning this layer range acts on it. Others
  // skip silently — no ack — so a sharded fleet produces exactly one ack per
  // range. Migration ranges are assumed to never cross worker boundaries, so
  // checking layer_begin is sufficient to identify the owner.
  if (!worker.ownsLayer(parsed->layer_begin)) {
    TT_LOG_DEBUG(
        "[bringup] skipping migration_id={}: layer range [{},{}) not owned",
        parsed->migration_id, parsed->layer_begin, parsed->layer_end);
    return;
  }

  TT_LOG_INFO(
      "[bringup] migration_id={} src_slot={} dst_slot={} "
      "layers=[{},{}) src_positions=[{},{}) dst_positions=[{},{})",
      parsed->migration_id, parsed->src_slot, parsed->dst_slot,
      parsed->layer_begin, parsed->layer_end, parsed->src_position_begin,
      parsed->src_position_end, parsed->dst_position_begin,
      parsed->dst_position_end);

  const tt::messaging::MigrationResponseMessage ack{
      .migration_id = parsed->migration_id,
      .status = tt::services::MigrationStatus::SUCCESSFUL,
  };
  std::string err;
  if (!ackProducer.send(tt::messaging::serialize(ack), &err)) {
    TT_LOG_ERROR("[bringup] ack send failed migration_id={}: {}",
                 parsed->migration_id, err);
  } else {
    TT_LOG_DEBUG("[bringup] acked migration_id={}", parsed->migration_id);
  }
}

void emitHeartbeatIfDue(const std::string& workerName,
                        const MooncakeMigrationWorker& worker,
                        std::chrono::steady_clock::time_point startTime,
                        std::chrono::steady_clock::time_point& nextHeartbeat) {
  const auto now = std::chrono::steady_clock::now();
  if (now < nextHeartbeat) return;
  const auto upSec =
      std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
  TT_LOG_INFO("[bringup] '{}' alive — {} peers, up {}s", workerName,
              worker.peers().size(), upSec);
  nextHeartbeat = now + std::chrono::seconds(K_HEARTBEAT_SEC);
}

// Single-threaded poll loop: drains migration requests from Kafka and emits a
// "still alive" heartbeat every K_HEARTBEAT_SEC. receive()'s timeout also paces
// stop-flag responsiveness (a SIGTERM is honored within K_KAFKA_POLL_MS).
void runMigrationLoop(const std::string& workerName,
                      MooncakeMigrationWorker& worker,
                      tt::messaging::IKafkaConsumer& requestConsumer,
                      tt::messaging::IKafkaProducer& ackProducer,
                      const std::atomic<bool>& stopRequested) {
  const auto startTime = std::chrono::steady_clock::now();
  auto nextHeartbeat = startTime + std::chrono::seconds(K_HEARTBEAT_SEC);

  while (!stopRequested.load()) {
    if (auto raw = requestConsumer.receive(K_KAFKA_POLL_MS); raw.has_value()) {
      handleMigrationRequest(*raw, worker, ackProducer);
    }
    emitHeartbeatIfDue(workerName, worker, startTime, nextHeartbeat);
  }
}

// No-Kafka path: hold open the Mooncake segment until SIGTERM. Receiver-role
// workers (e.g. decode peers in a prefill→decode topology) take this branch so
// they don't compete with prefill workers for Kafka requests.
void runIdleLoop(const std::string& workerName, MooncakeMigrationWorker& worker,
                 const std::atomic<bool>& stopRequested) {
  const auto startTime = std::chrono::steady_clock::now();
  auto nextHeartbeat = startTime + std::chrono::seconds(K_HEARTBEAT_SEC);

  while (!stopRequested.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(K_KAFKA_POLL_MS));
    emitHeartbeatIfDue(workerName, worker, startTime, nextHeartbeat);
  }
}

}  // namespace

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize("bringup-worker");

  WorkerConfig cli;
  if (!parseConfig(argc, argv, cli)) {
    usage();
    return 2;
  }
  installSignalHandlers();

  // Composition root: build the worker's collaborators (engine + discovery
  // service) and hand them to a worker that owns its own bring-up and teardown.
  // Kept inline because MooncakeMigrationWorker is non-movable (atomic member),
  // so it cannot be returned from a helper without heap-allocating.
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  auto discovery = std::make_shared<PeerDiscoveryService>(PeerDiscoveryConfig{
      K_DISCOVERY_POLL_INTERVAL_MS, cli.discovery_timeout_sec});
  MooncakeMigrationWorker worker{toWorkerConfig(cli), std::move(engine),
                                 std::move(discovery)};

  // Start the health surface BEFORE bring-up: discovery can block for up to
  // --discovery-timeout-sec, and a k8s liveness probe must succeed during that
  // window or the pod gets killed mid-bring-up. Readiness stays 503 until
  // bringUp() flips the worker Ready. Declared after `worker` so it is torn
  // down first (it borrows the worker's WorkerHealth).
  std::optional<WorkerHealthServer> healthServer;
  if (cli.health_port != 0) {
    healthServer.emplace(*worker.health(), cli.health_host, cli.health_port);
    if (!healthServer->start()) {
      TT_LOG_ERROR("[bringup] '{}' health server failed to bind {}:{}",
                   cli.name, cli.health_host, cli.health_port);
      return 1;
    }
  }

  // Pass the stop flag into bring-up too, so a SIGTERM/SIGINT during discovery
  // aborts promptly instead of blocking until the discovery timeout.
  if (!worker.bringUp(gStopRequested)) {
    TT_LOG_ERROR("[bringup] '{}' bring-up failed", cli.name);
    return 1;
  }

  if (cli.layer_end == 0) {
    TT_LOG_INFO("[bringup] '{}' owns all KV layers (no span configured)",
                cli.name);
  } else {
    TT_LOG_INFO("[bringup] '{}' owns KV layers [{}, {})", cli.name,
                cli.layer_start, cli.layer_end);
  }

  if (!cli.kafka_enabled) {
    TT_LOG_INFO(
        "[bringup] '{}' READY ({} peers); Kafka disabled — entering idle loop",
        cli.name, worker.peers().size());
    runIdleLoop(cli.name, worker, gStopRequested);
    TT_LOG_INFO("[bringup] '{}' stopping", cli.name);
    return 0;
  }

  const auto kafka = loadKafkaConfig();
  TT_LOG_INFO("[bringup] '{}' READY ({} peers); entering KV-migration loop",
              cli.name, worker.peers().size());
  TT_LOG_INFO(
      "[bringup] Kafka brokers={} request_topic={} ack_topic={} group={}",
      kafka.brokers, kafka.requestTopic, kafka.ackTopic, kafka.groupId);

  tt::messaging::KafkaConsumer requestConsumer{
      tt::messaging::KafkaConsumerConfig{
          .brokers = kafka.brokers,
          .topic = kafka.requestTopic,
          .group_id = kafka.groupId,
      }};
  tt::messaging::KafkaProducer ackProducer{tt::messaging::KafkaProducerConfig{
      .brokers = kafka.brokers,
      .topic = kafka.ackTopic,
  }};

  runMigrationLoop(cli.name, worker, requestConsumer, ackProducer,
                   gStopRequested);

  TT_LOG_INFO("[bringup] '{}' stopping", cli.name);
  // Stack-scope destructors run in reverse: Kafka clients close (the producer
  // flushes any in-flight ack on dtor; see KafkaProducer::~KafkaProducer),
  // then MooncakeMigrationWorker unregisters its published segment.
  return 0;
}
