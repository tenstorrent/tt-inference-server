// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// mooncake_kv_migration_worker: the single migration-worker binary that joins
// the Kafka trigger to the KV-migration data plane. One process, two roles:
//
//   --role prefill : consumes MigrationRequestMessages from Kafka and drives a
//                    real migration across the decode hosts via
//                    MooncakeMigrationExecutor → KvMigrationMultiHostSender,
//                    then publishes the ack. (KvMigrationWorker owns the loop.)
//
//   --role decode  : registers its KV mirror as a Mooncake segment and runs
//                    KvMigrationReceiverServer, answering the migration control
//                    protocol (prepareMirror / drain) for inbound migrations.
//
// This supersedes bringup_mooncake_worker (Kafka loop that only logged) and
// tt_kv_migration_consumer (StubMigrationExecutor): it is the first binary that
// actually moves KV on a Kafka trigger.
//
// Table source: each worker loads ONLY its own .pb. Prefill and decode swap
// tables once over the control channel (TABLE_EXCHANGE / #4295): prefill keeps
// the fleet decode table; decode stores the peer prefill table. TE/Mooncake
// moves KV bytes only — not table provisioning. The engine→worker
// handoff (engine_table_handoff) can later replace the local .pb behind the
// same IKvTable. Device IO: MultiDeviceUmd; FabricNode→ASIC chip resolution
// comes from an optional --device-map file, falling back to the placeholder
// (device & 0xFFFF) for a single-mesh host when no map is given.

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "config/defaults.hpp"
#include "messaging/kafka_consumer.hpp"
#include "messaging/kafka_producer.hpp"
#include "runtime/worker/kv_migration_worker.hpp"
#include "sockets/tcp_socket_transport.hpp"
#include "transport/device_map.hpp"  // DeviceMap (FabricNode -> UMD chip)
#include "transport/host_dram_storage_backend.hpp"
#include "transport/kv_migration_endpoints.hpp"
#include "transport/kv_migration_multi_host_sender.hpp"
#include "transport/kv_table_adapter.hpp"       // allHostLocations
#include "transport/kv_table_provisioning.hpp"  // loadKvTableFile
#include "transport/kv_table_view.hpp"          // IKvTable
#include "transport/mooncake_kv_receiver.hpp"
#include "transport/mooncake_migration_executor.hpp"
#include "transport/mooncake_transfer_engine.hpp"
#include "transport/multi_device_umd.hpp"
#include "transport/transfer_types.hpp"
#include "transport/umd_device_access.hpp"
#include "transport/worker_health.hpp"
#include "transport/worker_health_server.hpp"
#include "utils/logger.hpp"

namespace {

using namespace tt::transport;

constexpr int K_IDLE_POLL_MS = 100;

// How long the prefill worker waits for its decode control channels to finish
// their (asynchronous) TCP connect before it declares READY and starts
// consuming Kafka. On timeout it proceeds degraded (see runPrefill).
constexpr int K_CONNECT_TIMEOUT_MS = 30000;

// Control-channel receive budget must cover one TABLE_EXCHANGE of a large
// decode .pb (100–350+ MiB) at bring-up, not only tiny migrate messages.
constexpr int K_CONTROL_RECEIVE_TIMEOUT_MS = 300000;

// Default KV migration control port a decode binds when --control-port is
// omitted. The endpoint is published to the metadata service (see
// K_CONTROL_KEY_PREFIX), so prefill discovers host AND port from there rather
// than assuming this value; it only backstops mixed/old deploys that predate
// control-endpoint publishing.
constexpr uint16_t K_DEFAULT_CONTROL_PORT = 18650;

// Metadata key a worker publishes its KV control endpoint under, and a peer
// resolves it from: "kv_control/<name>" -> "<host>:<port>". This keeps every
// peer fact (segment, rpc host, and the control endpoint) in the one metadata
// service instead of a static convention.
constexpr const char* K_CONTROL_KEY_PREFIX = "kv_control/";

constexpr int K_DISCOVERY_TIMEOUT_MS = 30000;
constexpr int K_DISCOVERY_POLL_MS = 1000;

std::atomic<bool> gStop{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "gStop must be lock-free to be signal-safe");
void onSignal(int /*sig*/) { gStop.store(true, std::memory_order_relaxed); }

std::string envOr(const char* key, const char* fallback) {
  const char* v = std::getenv(key);
  return (v != nullptr && *v != '\0') ? std::string{v} : std::string{fallback};
}

enum class Role { PREFILL, DECODE };

struct WorkerConfig {
  Role role = Role::PREFILL;
  std::string metadata_uri;     // Mooncake metadata service (or P2PHANDSHAKE).
  std::string name;             // this worker's Mooncake server/segment name.
  std::string host;             // this node's host tag in the table.
  std::string device_map_path;  // FabricNode->UMD chip map (both roles); empty
                                // => placeholder (device & 0xFFFF).

  // Peers (any role — a worker is just a migration worker with a peer list).
  // Static peer control endpoints (NAME=host:port). An explicit entry always
  // wins over discovery for the same NAME.
  std::unordered_map<std::string, KvControlChannelConnector::Endpoint> peers;
  // Peers to DISCOVER via the metadata service, by the peer's logical name (its
  // --name). Both the routable host and the control port come from the peer's
  // published "kv_control/<name>" entry. So each --peer TAG must match the
  // corresponding peer's --name (e.g. "decode-0"); with a metadata server that
  // TAG can be logical (no IP), which is the whole point. The prefill acts on
  // these (opens control channels); a decode only resolves + holds them.
  std::vector<std::string> discover_peers;
  // Fallback control port, used only for a discovered peer whose control
  // endpoint is not (yet) published — otherwise the port comes from metadata.
  uint16_t peer_control_port = 0;

  // prefill tables:
  std::string prefill_table_path;
  std::string decode_table_path;

  // decode:
  std::string table_path;  // this decode node's table.
  std::string segment;     // advertised segment (defaults to `name`).
  uint16_t control_port = 0;

  // HTTP health surface (/healthz, /readyz, /metrics). 0 == disabled so local
  // runs and the e2e stay port-free unless a port is explicitly requested.
  uint16_t health_port = 0;
  std::string health_host = "0.0.0.0";
};

void usage() {
  std::cerr
      << "usage: mooncake_kv_migration_worker --metadata URI --name NAME "
         "--host HOST_TAG [--role prefill|decode]\n"
         "  --role is inferred from NAME's prefix (prefill*/decode*) when "
         "omitted.\n"
         "  peers (any role): --peer NAME (repeatable; NAME==a peer's --name, "
         "its control host:port is discovered via metadata) and/or\n"
         "           --peer-control NAME=host:port (repeatable, static)\n"
         "           [--peer-control-port N]  fallback control port for a "
         "discovered peer that hasn't published its endpoint (default 18650).\n"
         "           The prefill (sender) opens a control channel to each "
         "peer; TABLE_EXCHANGE swaps tables once (prefill↔decode), then "
         "migrations use the same channels.\n"
         "  prefill: --prefill-table P.pb (+ >=1 peer); decode table comes "
         "from control TABLE_EXCHANGE (optional --decode-table fallback)\n"
         "  decode:  --table D.pb [--control-port N] (default 18650) "
         "[--segment NAME]; stores peer prefill table on TABLE_EXCHANGE\n"
         "  both:    [--device-map FILE]  ('mesh chip umd' per line; needed "
         "when this host's table spans multiple meshes)\n"
         "  both:    [--health-port N] [--health-host HOST]  serve "
         "/healthz /readyz /metrics (0=off, default off; host default "
         "0.0.0.0)\n"
         "  Kafka (prefill) via env: KAFKA_BROKERS, "
         "KAFKA_MIGRATION_REQUEST_TOPIC, KAFKA_MIGRATION_ACK_TOPIC, "
         "KAFKA_GROUP_ID\n";
}

// Parse "NAME=host:port" into (logical name, endpoint).
bool parsePeer(const std::string& spec, std::string& name,
               KvControlChannelConnector::Endpoint& ep) {
  const auto eq = spec.find('=');
  const auto colon = spec.rfind(':');
  if (eq == std::string::npos || colon == std::string::npos || colon < eq) {
    return false;
  }
  name = spec.substr(0, eq);
  ep.host = spec.substr(eq + 1, colon - eq - 1);
  try {
    ep.port = static_cast<uint16_t>(std::stoi(spec.substr(colon + 1)));
  } catch (...) {
    return false;
  }
  return !name.empty() && !ep.host.empty() && ep.port != 0;
}

// Parse "host:port" (a published control endpoint) into an Endpoint.
bool parseEndpoint(const std::string& spec,
                   KvControlChannelConnector::Endpoint& ep) {
  const auto colon = spec.rfind(':');
  if (colon == std::string::npos || colon == 0) return false;
  ep.host = spec.substr(0, colon);
  try {
    ep.port = static_cast<uint16_t>(std::stoi(spec.substr(colon + 1)));
  } catch (...) {
    return false;
  }
  return !ep.host.empty() && ep.port != 0;
}

// The host part of the engine's advertised "host:port" server name — the same
// routable host Mooncake registered in rpc_meta, so a peer reaches our control
// server at exactly the address it reaches our data plane.
std::string hostOf(const std::string& serverName) {
  const auto colon = serverName.rfind(':');
  return colon == std::string::npos ? serverName : serverName.substr(0, colon);
}

bool parseConfig(int argc, char** argv, WorkerConfig& cfg) {
  bool roleGiven = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string& dst) {
      if (i + 1 >= argc) return false;
      dst = argv[++i];
      return true;
    };
    std::string v;
    if (a == "--role" && next(v)) {
      if (v == "prefill") {
        cfg.role = Role::PREFILL;
      } else if (v == "decode") {
        cfg.role = Role::DECODE;
      } else {
        std::cerr << "--role must be prefill|decode\n";
        return false;
      }
      roleGiven = true;
      continue;
    }
    if (a == "--metadata" && next(cfg.metadata_uri)) continue;
    if (a == "--name" && next(cfg.name)) continue;
    if (a == "--host" && next(cfg.host)) continue;
    if (a == "--prefill-table" && next(cfg.prefill_table_path)) continue;
    if (a == "--decode-table" && next(cfg.decode_table_path)) continue;
    if (a == "--table" && next(cfg.table_path)) continue;
    if (a == "--device-map" && next(cfg.device_map_path)) continue;
    if (a == "--segment" && next(cfg.segment)) continue;
    if (a == "--control-port" && next(v)) {
      try {
        cfg.control_port = static_cast<uint16_t>(std::stoi(v));
      } catch (...) {
        std::cerr << "--control-port invalid: " << v << "\n";
        return false;
      }
      continue;
    }
    if (a == "--peer-control" && next(v)) {
      std::string name;
      KvControlChannelConnector::Endpoint ep;
      if (!parsePeer(v, name, ep)) {
        std::cerr << "--peer-control must be NAME=host:port, got: " << v
                  << "\n";
        return false;
      }
      cfg.peers[name] = ep;
      continue;
    }
    if (a == "--peer" && next(v)) {
      cfg.discover_peers.push_back(v);
      continue;
    }
    if (a == "--peer-control-port" && next(v)) {
      try {
        const int port = std::stoi(v);
        if (port <= 0 || port > 65535) throw std::out_of_range("range");
        cfg.peer_control_port = static_cast<uint16_t>(port);
      } catch (...) {
        std::cerr << "--peer-control-port must be 1..65535, got: " << v << "\n";
        return false;
      }
      continue;
    }
    if (a == "--health-port" && next(v)) {
      try {
        const int port = std::stoi(v);
        if (port < 0 || port > 65535) throw std::out_of_range("range");
        cfg.health_port = static_cast<uint16_t>(port);
      } catch (...) {
        std::cerr << "--health-port must be 0..65535, got: " << v << "\n";
        return false;
      }
      continue;
    }
    if (a == "--health-host" && next(cfg.health_host)) continue;
    std::cerr << "unknown/incomplete arg: " << a << "\n";
    return false;
  }

  if (cfg.metadata_uri.empty() || cfg.name.empty() || cfg.host.empty()) {
    std::cerr << "--metadata, --name and --host are required\n";
    return false;
  }
  // Infer the role from the name prefix when --role was omitted (the deploy
  // names workers prefill-N / decode-N), so the common launch needs no --role.
  if (!roleGiven) {
    if (cfg.name.rfind("prefill", 0) == 0) {
      cfg.role = Role::PREFILL;
    } else if (cfg.name.rfind("decode", 0) == 0) {
      cfg.role = Role::DECODE;
    } else {
      std::cerr << "--role is required when --name does not start with "
                   "'prefill' or 'decode'\n";
      return false;
    }
  }
  // One shared control port per host: decode listens here, prefill dials it for
  // discovered peers. Fixed (not per-peer), matching the bringup-era deploy.
  if (cfg.control_port == 0) cfg.control_port = K_DEFAULT_CONTROL_PORT;
  if (cfg.peer_control_port == 0)
    cfg.peer_control_port = K_DEFAULT_CONTROL_PORT;

  if (cfg.role == Role::PREFILL &&
      (cfg.prefill_table_path.empty() ||
       (cfg.peers.empty() && cfg.discover_peers.empty() &&
        cfg.decode_table_path.empty()))) {
    std::cerr << "prefill needs --prefill-table and either peers "
                 "(--peer / --peer-control) for TABLE_EXCHANGE, or "
                 "--decode-table as a local fallback\n";
    return false;
  }
  if (cfg.role == Role::DECODE && cfg.table_path.empty()) {
    std::cerr << "decode needs --table\n";
    return false;
  }
  // NB: segment defaults to the engine's real local server name at runtime
  // (runDecode), not cfg.name — under P2PHANDSHAKE the RPC port is
  // auto-assigned so engine->localServerName() != cfg.name, and the sender must
  // open the engine's actual segment. Only an explicit --segment overrides it.
  return true;
}

std::shared_ptr<MooncakeTransferEngine> makeEngine(const WorkerConfig& cfg) {
  auto engine = std::make_shared<MooncakeTransferEngine>(
      std::make_shared<HostDramStorageBackend>());
  EngineConfig ec;
  ec.metadata_uri = cfg.metadata_uri;
  ec.local_server_name = cfg.name;
  ec.protocol = TransportProtocol::TCP;
  if (!engine->init(ec)) {
    TT_LOG_ERROR("[worker] engine init failed (metadata={})", cfg.metadata_uri);
    return nullptr;
  }
  return engine;
}

// Load a 'mesh chip umd' device map (the same format the e2e harness uses).
// Empty path => empty map => buildDeviceIo falls back to the placeholder.
DeviceMap loadDeviceMapFile(const std::string& path) {
  DeviceMap dm;
  if (path.empty()) return dm;
  std::ifstream f(path);
  if (!f.good()) {
    TT_LOG_WARN(
        "[worker] cannot open --device-map {}; using placeholder chip ids",
        path);
    return dm;
  }
  uint32_t mesh = 0, chip = 0;
  uint64_t umd = 0;
  while (f >> mesh >> chip >> umd) dm.set(FabricNode{mesh, chip}, umd);
  TT_LOG_INFO("[worker] device-map: {} entries from {}", dm.size(), path);
  return dm;
}

// One UmdDeviceAccess per device this host owns in `table`. Chip resolution
// uses `deviceMap` (the interface the engine fills; the file-based map is the
// same contract) and falls back to the encodeDevice low bits (placeholder, as
// in the e2e harness) for any device the map doesn't cover. The placeholder is
// only correct when the host's table is single-mesh; a multi-mesh host (its KV
// aliases chip ids across meshes) needs the map or its replicas collide.
std::unique_ptr<MultiDeviceUmd> buildDeviceIo(const IKvTable& table,
                                              const std::string& host,
                                              const DeviceMap& deviceMap) {
  auto umd = std::make_unique<MultiDeviceUmd>();
  for (const auto& loc : allHostLocations(table, host)) {
    if (!umd->hasDevice(loc.device)) {
      const auto mapped = deviceMap.umdChip(loc.device);
      const int chip = mapped ? static_cast<int>(*mapped)
                              : static_cast<int>(loc.device & 0xFFFFu);
      umd->addDevice(loc.device, std::make_shared<UmdDeviceAccess>(chip));
    }
  }
  return umd;
}

std::shared_ptr<tt::sockets::ISocketTransport> makeClientTransport(
    const KvControlChannelConnector::Endpoint& ep) {
  auto t = std::make_shared<tt::sockets::TcpSocketTransport>();
  if (!t->initializeAsClient(ep.host, ep.port)) return nullptr;
  t->start();
  return t;
}

std::shared_ptr<tt::sockets::ISocketTransport> makeServerTransport(
    uint16_t port) {
  auto t = std::make_shared<tt::sockets::TcpSocketTransport>();
  if (!t->initializeAsServer(port)) return nullptr;
  // Do NOT start() here — KvMigrationReceiverServer installs the multi-accept
  // handler first, then start()s, so every prefill gets its own session.
  return t;
}

// Bring up the optional HTTP health surface (/healthz /readyz /metrics) before
// the worker's own bring-up, so a k8s liveness probe succeeds during the
// (possibly slow) connect/register phase while /readyz stays 503 until we flip
// the worker Ready. A port of 0 means disabled (success, no server). @p out
// must outlive @p health's server; both are the caller's stack locals so the
// server is torn down first. Returns false only on a real bind failure.
bool startHealthServer(WorkerHealth& health, const WorkerConfig& cfg,
                       std::optional<WorkerHealthServer>& out) {
  if (cfg.health_port == 0) return true;
  out.emplace(health, cfg.health_host, cfg.health_port);
  if (!out->start()) {
    TT_LOG_ERROR("[worker] '{}' health server failed to bind {}:{}", cfg.name,
                 cfg.health_host, cfg.health_port);
    return false;
  }
  TT_LOG_INFO("[worker] '{}' health surface on {}:{}", cfg.name,
              cfg.health_host, out->port());
  return true;
}

// One attempt to resolve a single peer's control endpoint. Primary source is
// the peer's published "kv_control/<name>" (host AND port); the fallback is
// resolveServerName's rpc_meta host paired with the fixed peer_control_port,
// for a peer that registered its data plane but hasn't published a control
// endpoint (mixed/old deploy). Returns false if neither source resolves it yet.
bool resolveOnePeer(ITransferEngine& engine, const WorkerConfig& cfg,
                    const std::string& name,
                    KvControlChannelConnector::Endpoint& ep) {
  if (auto endpoint =
          engine.lookupMetadata(std::string(K_CONTROL_KEY_PREFIX) + name)) {
    if (parseEndpoint(*endpoint, ep)) {
      TT_LOG_INFO("[worker] discovered peer '{}' -> control {}:{} (metadata)",
                  name, ep.host, ep.port);
      return true;
    }
    TT_LOG_WARN(
        "[worker] peer '{}' published a malformed control endpoint '{}'; "
        "ignoring",
        name, *endpoint);
  }
  const std::string host = engine.resolveServerName(name);
  if (host.empty()) return false;
  ep = KvControlChannelConnector::Endpoint{host, cfg.peer_control_port};
  TT_LOG_INFO(
      "[worker] discovered peer '{}' -> control {}:{} (rpc_meta host + fixed "
      "port; control endpoint not published)",
      name, host, cfg.peer_control_port);
  return true;
}

// Resolve every peer this worker was given (role-agnostic — a worker is just a
// migration worker with a peer list) to its control endpoint via the metadata
// service, retrying until all resolve or the timeout elapses. Static
// --peer-control entries win and skip resolution; a SIGTERM (@p stop) aborts
// the wait promptly. Returns the name->endpoint map, possibly partial on
// timeout.
std::unordered_map<std::string, KvControlChannelConnector::Endpoint>
resolvePeers(ITransferEngine& engine, const WorkerConfig& cfg,
             const std::atomic<bool>& stop) {
  auto peers = cfg.peers;
  std::vector<std::string> pending;
  for (const auto& name : cfg.discover_peers)
    if (peers.count(name) == 0) pending.push_back(name);

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(K_DISCOVERY_TIMEOUT_MS);
  while (!pending.empty() && !stop.load()) {
    std::vector<std::string> unresolved;
    for (const auto& name : pending) {
      KvControlChannelConnector::Endpoint ep;
      if (resolveOnePeer(engine, cfg, name, ep)) {
        peers[name] = ep;
      } else {
        unresolved.push_back(name);
      }
    }
    pending.swap(unresolved);
    if (pending.empty() || std::chrono::steady_clock::now() >= deadline) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
  }
  for (const auto& name : pending)
    TT_LOG_WARN(
        "[worker] peer '{}' unresolved after {}ms; its slice fails until it "
        "registers and is re-resolved",
        name, K_DISCOVERY_TIMEOUT_MS);
  return peers;
}

// Pull the one fleet decode table from the first connected decode control
// channel (TABLE_EXCHANGE). All decodes share the same .pb — one successful
// exchange is enough.
std::shared_ptr<const IKvTable> provisionDecodeTableFromControl(
    const std::unordered_map<std::string, KvControlChannel*>& channels,
    const std::vector<uint8_t>& prefillBlob) {
  for (const auto& [host, channel] : channels) {
    if (channel == nullptr || !channel->isConnected()) {
      continue;
    }
    auto table =
        provisionPeerTable(*channel, TableExchangeRole::Sender, prefillBlob);
    if (table) {
      TT_LOG_INFO(
          "[worker] got decode table via control TABLE_EXCHANGE with '{}' "
          "({} B local prefill blob)",
          host, prefillBlob.size());
      return table;
    }
    TT_LOG_WARN(
        "[worker] TABLE_EXCHANGE with '{}' failed; trying next connected peer",
        host);
  }
  return nullptr;
}

int runPrefill(const WorkerConfig& cfg) {
  // Declared before the health server so the server (which borrows this) is
  // destroyed first; liveness is up immediately, readiness stays false until
  // the connect barrier below flips us Ready.
  WorkerHealth health(cfg.name);
  std::optional<WorkerHealthServer> healthServer;
  if (!startHealthServer(health, cfg, healthServer)) return 1;

  auto engine = makeEngine(cfg);
  if (!engine) return 1;

  auto prefill = loadKvTableFile(cfg.prefill_table_path);
  if (!prefill) {
    TT_LOG_ERROR("[worker] failed to load prefill table");
    return 1;
  }

  const DeviceMap deviceMap = loadDeviceMapFile(cfg.device_map_path);
  auto device = buildDeviceIo(*prefill->table, cfg.host, deviceMap);

  // Discover every decode peer's control endpoint through the metadata service
  // (host AND port from its published "kv_control/<name>"), retrying until they
  // resolve — this is what lets --peer be a logical tag like "decode-0" instead
  // of a hardcoded endpoint. The prefill is the sender, so it ACTS on the
  // peers: openChannels() below creates a control channel to each (the TCP
  // connect then runs asynchronously in each transport's background loop).
  auto peers = resolvePeers(*engine, cfg, gStop);

  KvControlChannelConnector connector(
      peers, makeClientTransport,
      std::chrono::milliseconds(K_CONTROL_RECEIVE_TIMEOUT_MS));
  if (!connector.openChannels()) {
    TT_LOG_WARN(
        "[worker] not every decode peer got a transport; involved-but-missing "
        "hosts will fail their slice");
  }

  // Startup barrier: wait for control channels before TABLE_EXCHANGE / Kafka.
  const std::size_t total = connector.channelCount();
  const std::size_t connected =
      connector.awaitConnected(std::chrono::milliseconds(K_CONNECT_TIMEOUT_MS));
  if (connected < total) {
    TT_LOG_WARN(
        "[worker] only {}/{} decode control channels connected within {}ms; "
        "migrations to unconnected hosts will fail until they connect",
        connected, total, K_CONNECT_TIMEOUT_MS);
  }

  // One fleet decode table: pull once from any connected decode, or disk
  // fallback when running without peers.
  std::shared_ptr<const IKvTable> decodeTable;
  const bool wantsExchange = !cfg.discover_peers.empty() || !cfg.peers.empty();
  if (wantsExchange) {
    decodeTable =
        provisionDecodeTableFromControl(connector.channels(), prefill->blob);
    if (!decodeTable) {
      TT_LOG_ERROR(
          "[worker] control TABLE_EXCHANGE failed (need >=1 connected decode)");
      return 1;
    }
  } else if (!cfg.decode_table_path.empty()) {
    auto decode = loadKvTableFile(cfg.decode_table_path);
    if (!decode) {
      TT_LOG_ERROR("[worker] failed to load --decode-table fallback");
      return 1;
    }
    decodeTable = decode->table;
  } else {
    TT_LOG_ERROR(
        "[worker] prefill needs peers for TABLE_EXCHANGE or --decode-table");
    return 1;
  }

  KvMigrationMultiHostSender sender(engine, *device, prefill->table,
                                    decodeTable, cfg.host, connector.channels(),
                                    &health);
  auto executor = std::make_unique<MooncakeMigrationExecutor>(sender);

  const std::string brokers =
      envOr("KAFKA_BROKERS", tt::config::defaults::KAFKA_BROKERS);
  const std::string reqTopic =
      envOr("KAFKA_MIGRATION_REQUEST_TOPIC",
            tt::config::defaults::KAFKA_MIGRATION_REQUEST_TOPIC);
  const std::string ackTopic =
      envOr("KAFKA_MIGRATION_ACK_TOPIC",
            tt::config::defaults::KAFKA_MIGRATION_ACK_TOPIC);
  const std::string group =
      envOr("KAFKA_GROUP_ID", tt::config::defaults::KAFKA_GROUP_ID);

  auto consumer = std::make_unique<tt::messaging::KafkaConsumer>(
      tt::messaging::KafkaConsumerConfig{
          .brokers = brokers, .topic = reqTopic, .group_id = group});
  auto producer = std::make_unique<tt::messaging::KafkaProducer>(
      tt::messaging::KafkaProducerConfig{.brokers = brokers,
                                         .topic = ackTopic});

  tt::worker::KvMigrationWorker worker(std::move(consumer), std::move(producer),
                                       std::move(executor));

  // Readiness gate: only after connect + table provision do we flip Ready.
  if (connected == 0 && total > 0) {
    health.setProcessHealthy(true);
    TT_LOG_WARN(
        "[worker] prefill '{}' has 0/{} decode channels connected; staying "
        "not-ready until a peer connects",
        cfg.name, total);
  } else {
    health.setLifecycle(WorkerLifecycle::Ready);
  }

  TT_LOG_INFO(
      "[worker] prefill '{}' READY: {}/{} decode channels connected, "
      "brokers={} req={} ack={}",
      cfg.name, connected, total, brokers, reqTopic, ackTopic);
  worker.start();
  while (!gStop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(K_IDLE_POLL_MS));
  }
  worker.stop();
  health.setLifecycle(WorkerLifecycle::ShuttingDown);
  TT_LOG_INFO("[worker] prefill '{}' stopping", cfg.name);
  return 0;
}

int runDecode(const WorkerConfig& cfg) {
  WorkerHealth health(cfg.name);
  std::optional<WorkerHealthServer> healthServer;
  if (!startHealthServer(health, cfg, healthServer)) return 1;

  auto engine = makeEngine(cfg);
  if (!engine) return 1;

  auto decode = loadKvTableFile(cfg.table_path);
  if (!decode) {
    TT_LOG_ERROR("[worker] failed to load decode table {}", cfg.table_path);
    return 1;
  }

  const DeviceMap deviceMap = loadDeviceMapFile(cfg.device_map_path);
  auto device = buildDeviceIo(*decode->table, cfg.host, deviceMap);

  // Segment name the sender opens for the data plane. With a metadata service
  // the mirror is registered under — and resolvable by — the worker's LOGICAL
  // name (cfg.name). Only P2PHANDSHAKE falls back to the engine's live IP:port.
  // An explicit --segment always overrides.
  std::string segment = cfg.segment;
  if (segment.empty()) {
    segment = (cfg.metadata_uri == "P2PHANDSHAKE") ? engine->localServerName()
                                                   : cfg.name;
  }
  MooncakeKvReceiver receiver(engine, *device, decode->table, cfg.host,
                              segment);
  if (!receiver.registered()) {
    TT_LOG_ERROR("[worker] decode '{}' failed to register mirror segment '{}'",
                 cfg.name, segment);
    return 1;
  }

  // Control server stores peer prefill .pb on TABLE_EXCHANGE, replies with
  // this decode .pb, then serves migrate Begin/Done. Long receive timeout
  // covers large table provisioning.
  KvMigrationReceiverServer server(
      cfg.control_port, makeServerTransport, receiver, decode->blob,
      std::chrono::milliseconds(K_CONTROL_RECEIVE_TIMEOUT_MS));
  if (!server.start()) {
    TT_LOG_ERROR("[worker] decode '{}' failed to start control server on :{}",
                 cfg.name, cfg.control_port);
    return 1;
  }

  const std::string controlEndpoint = hostOf(engine->localServerName()) + ":" +
                                      std::to_string(cfg.control_port);
  const std::string controlKey = std::string(K_CONTROL_KEY_PREFIX) + cfg.name;
  if (engine->publishMetadata(controlKey, controlEndpoint)) {
    TT_LOG_INFO("[worker] decode '{}' published control endpoint {} -> {}",
                cfg.name, controlKey, controlEndpoint);
  } else if (cfg.metadata_uri != "P2PHANDSHAKE") {
    TT_LOG_WARN(
        "[worker] decode '{}' could not publish control endpoint to metadata "
        "({}); prefill must fall back to the fixed control port",
        cfg.name, controlKey);
  }

  health.setLifecycle(WorkerLifecycle::Ready);
  TT_LOG_INFO("[worker] decode '{}' READY: segment={} control_port={}",
              cfg.name, segment, cfg.control_port);
  while (!gStop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(K_IDLE_POLL_MS));
  }
  server.stop();
  health.setLifecycle(WorkerLifecycle::ShuttingDown);
  TT_LOG_INFO("[worker] decode '{}' stopping", cfg.name);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize("kv-migration-worker");

  WorkerConfig cfg;
  if (!parseConfig(argc, argv, cfg)) {
    usage();
    return 2;
  }
  std::signal(SIGTERM, onSignal);
  std::signal(SIGINT, onSignal);

  return cfg.role == Role::PREFILL ? runPrefill(cfg) : runDecode(cfg);
}
