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
// Table source: each worker loads its KV .pb from disk (--prefill-table /
// --table; production path is typically under /tmp from the engine export).
// DeviceMap comes from a localhost socket (--engine-handoff-port) or a
// --device-map file; socket wins if both are set. Prefill and decode then swap
// tables over the control channel (TABLE_EXCHANGE / #4295). TE/Mooncake moves
// KV bytes only. MultiDeviceUmd uses the map; an empty map may use the
// single-mesh placeholder, but a non-empty map with a missing entry is fatal.
// KV_MIGRATION_MODE=dry-run keeps discovery, control-table exchange, health,
// and Kafka active without opening devices or moving KV data.

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
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
#include "runtime/worker/stub_migration_executor.hpp"
#include "sockets/tcp_socket_transport.hpp"
#include "transport/device_map.hpp"  // DeviceMap (FabricNode -> UMD chip)
#include "transport/engine_table_resolve.hpp"
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

// TABLE_EXCHANGE uses kDefaultTableExchangeTimeout (large .pb blobs). Migrate
// MirrorReady/Ack stay on the shorter channel receiveTimeout below.
constexpr int K_MIGRATE_CONTROL_TIMEOUT_MS = 30000;

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

// Peer discovery: poll metadata until every configured peer resolves, or
// SIGTERM. K_DISCOVERY_TIMEOUT_MS is only the WARN heartbeat cadence (k8s
// readiness stays 503; liveness stays up) — we never give up and proceed
// degraded. Same cadence for the TCP connect barrier after openChannels().
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
enum class MigrationMode { DEVICE, DRY_RUN };

struct WorkerConfig {
  Role role = Role::PREFILL;
  MigrationMode migrationMode = MigrationMode::DEVICE;
  std::string metadata_uri;     // Mooncake metadata service (or P2PHANDSHAKE).
  std::string name;             // this worker's Mooncake server/segment name.
  std::string host;             // this node's host tag in the table.
  std::string device_map_path;  // FabricNode->UMD file fallback; empty => empty
                                // map (placeholder only if single-mesh).
                                // Ignored when engine_handoff_port != 0.
  // When non-zero, listen for one DeviceMap handoff after loading the .pb file.
  uint16_t engine_handoff_port = 0;

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
         "  both:    [--engine-handoff-port N]  after loading the .pb, listen "
         "for a DeviceMap handoff (preferred over --device-map)\n"
         "  both:    [--device-map FILE]  ('mesh chip umd' per line; file "
         "fallback when handoff port is unset; needed for multi-mesh)\n"
         "  both:    [--health-port N] [--health-host HOST]  serve "
         "/healthz /readyz /metrics (0=off, default off; host default "
         "0.0.0.0)\n"
         "  Kafka (prefill) via env: KAFKA_BROKERS, "
         "KAFKA_MIGRATION_REQUEST_TOPIC, KAFKA_MIGRATION_ACK_TOPIC, "
         "KAFKA_GROUP_ID\n"
         "  worker mode via env: KV_MIGRATION_MODE=device|dry-run "
         "(default device; dry-run keeps discovery, table exchange, health, "
         "and Kafka but does not open devices or move KV data)\n";
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
  const std::string migrationMode = envOr("KV_MIGRATION_MODE", "device");
  if (migrationMode == "dry-run") {
    cfg.migrationMode = MigrationMode::DRY_RUN;
  } else if (migrationMode != "device") {
    std::cerr << "KV_MIGRATION_MODE must be device|dry-run, got: "
              << migrationMode << "\n";
    return false;
  }

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
    if (a == "--engine-handoff-port" && next(v)) {
      try {
        const int port = std::stoi(v);
        if (port <= 0 || port > 65535) throw std::out_of_range("range");
        cfg.engine_handoff_port = static_cast<uint16_t>(port);
      } catch (...) {
        std::cerr << "--engine-handoff-port must be 1..65535, got: " << v
                  << "\n";
        return false;
      }
      continue;
    }
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

  if (cfg.role == Role::PREFILL) {
    if (cfg.prefill_table_path.empty()) {
      std::cerr
          << "prefill needs --prefill-table (engine .pb path, e.g. /tmp)\n";
      return false;
    }
    if (cfg.peers.empty() && cfg.discover_peers.empty() &&
        cfg.decode_table_path.empty()) {
      std::cerr << "prefill needs peers (--peer / --peer-control) for "
                   "TABLE_EXCHANGE, or --decode-table as a local fallback\n";
      return false;
    }
  }
  if (cfg.role == Role::DECODE && cfg.table_path.empty()) {
    std::cerr << "decode needs --table (engine .pb path, e.g. /tmp)\n";
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

// One UmdDeviceAccess per device this host owns in `table`.
// - Non-empty deviceMap: every local device must be present (hard error if
// not).
// - Empty deviceMap: single-mesh hosts may use the & 0xFFFF placeholder; a
//   multi-mesh host with no map is a hard error (cross-mesh collision).
std::unique_ptr<MultiDeviceUmd> buildDeviceIo(const IKvTable& table,
                                              const std::string& host,
                                              const DeviceMap& deviceMap) {
  auto umd = std::make_unique<MultiDeviceUmd>();
  std::optional<uint32_t> seenMesh;
  bool isMultiMesh = false;
  for (const auto& loc : allHostLocations(table, host)) {
    const uint32_t mesh = static_cast<uint32_t>(loc.device >> 16);
    if (!seenMesh) {
      seenMesh = mesh;
    } else if (*seenMesh != mesh) {
      isMultiMesh = true;
    }
  }
  if (deviceMap.empty() && isMultiMesh) {
    TT_LOG_ERROR(
        "[worker] host '{}' table spans multiple meshes but DeviceMap is "
        "empty — refusing & 0xFFFF placeholder (cross-mesh collision). Pass "
        "--engine-handoff-port or --device-map",
        host);
    return nullptr;
  }

  for (const auto& loc : allHostLocations(table, host)) {
    if (umd->hasDevice(loc.device)) continue;
    const auto mapped = deviceMap.umdChip(loc.device);
    int chip = 0;
    if (mapped) {
      chip = static_cast<int>(*mapped);
    } else if (deviceMap.empty()) {
      chip = static_cast<int>(loc.device & 0xFFFFu);
    } else {
      TT_LOG_ERROR(
          "[worker] DeviceMap missing entry for device {} (mesh={} chip={}) "
          "on host '{}' — refusing & 0xFFFF placeholder once a map is in play",
          loc.device, loc.device >> 16, loc.device & 0xFFFFu, host);
      return nullptr;
    }
    umd->addDevice(loc.device, std::make_shared<UmdDeviceAccess>(chip));
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

// Same unstarted listen transport for engine handoff (enableMultiAccept first).
std::shared_ptr<tt::sockets::ISocketTransport> makeHandoffListenTransport(
    uint16_t port) {
  return makeServerTransport(port);
}

void warnIgnoredDeviceMapFile(const WorkerConfig& cfg) {
  if (cfg.engine_handoff_port == 0 || cfg.device_map_path.empty()) return;
  TT_LOG_WARN(
      "[worker] --engine-handoff-port={} set; ignoring --device-map {} (socket "
      "DeviceMap wins; .pb still loaded from --prefill-table/--table)",
      cfg.engine_handoff_port, cfg.device_map_path);
}

std::optional<ResolvedEngineTables> resolveWorkerTables(
    const WorkerConfig& cfg, bool includeDeviceMap = true) {
  if (!includeDeviceMap) {
    if (cfg.engine_handoff_port != 0 || !cfg.device_map_path.empty()) {
      TT_LOG_WARN(
          "[worker] dry-run mode ignores --engine-handoff-port and "
          "--device-map");
    }
    const std::string& tablePath =
        (cfg.role == Role::PREFILL) ? cfg.prefill_table_path : cfg.table_path;
    return resolveEngineTables(0, makeHandoffListenTransport, tablePath, "",
                               gStop);
  }

  warnIgnoredDeviceMapFile(cfg);
  const std::string& tablePath =
      (cfg.role == Role::PREFILL) ? cfg.prefill_table_path : cfg.table_path;
  return resolveEngineTables(cfg.engine_handoff_port,
                             makeHandoffListenTransport, tablePath,
                             cfg.device_map_path, gStop);
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
//
// INFO only when the endpoint is new or changed vs @p previousEp — mesh watch
// re-resolves every poll and must not spam "discovered peer" for sticky peers.
bool resolveOnePeer(
    ITransferEngine& engine, const WorkerConfig& cfg, const std::string& name,
    KvControlChannelConnector::Endpoint& ep,
    const KvControlChannelConnector::Endpoint* previousEp = nullptr) {
  auto logIfChanged = [&](const char* source) {
    if (previousEp != nullptr && *previousEp == ep) {
      return;
    }
    TT_LOG_INFO("[worker] discovered peer '{}' -> control {}:{} ({})", name,
                ep.host, ep.port, source);
  };

  if (auto endpoint =
          engine.lookupMetadata(std::string(K_CONTROL_KEY_PREFIX) + name)) {
    if (parseEndpoint(*endpoint, ep)) {
      logIfChanged("metadata");
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
  logIfChanged("rpc_meta host + fixed port; control endpoint not published");
  return true;
}

// Resolve every configured peer before bring-up continues. Static
// --peer-control entries win. Blocks until pending is empty or @p stop fires;
// logs a WARN heartbeat every K_DISCOVERY_TIMEOUT_MS while still waiting.
// Returns nullopt if stopped with peers still unresolved (caller should exit
// without becoming Ready).
std::optional<
    std::unordered_map<std::string, KvControlChannelConnector::Endpoint>>
resolveAllPeers(ITransferEngine& engine, const WorkerConfig& cfg,
                const std::atomic<bool>& stop) {
  auto resolved = cfg.peers;
  std::vector<std::string> pending;
  for (const auto& name : cfg.discover_peers) {
    if (resolved.count(name) == 0) pending.push_back(name);
  }
  if (pending.empty()) return resolved;

  auto lastWarn = std::chrono::steady_clock::now();
  TT_LOG_INFO("[worker] waiting for {} peer(s) to publish control endpoints",
              pending.size());
  while (!pending.empty() && !stop.load()) {
    std::vector<std::string> unresolved;
    for (const auto& name : pending) {
      KvControlChannelConnector::Endpoint ep;
      if (resolveOnePeer(engine, cfg, name, ep)) {
        resolved[name] = ep;
      } else {
        unresolved.push_back(name);
      }
    }
    pending.swap(unresolved);
    if (pending.empty()) break;

    const auto now = std::chrono::steady_clock::now();
    if (now - lastWarn >= std::chrono::milliseconds(K_DISCOVERY_TIMEOUT_MS)) {
      std::string missing;
      for (const auto& name : pending) {
        if (!missing.empty()) missing += ", ";
        missing += name;
      }
      TT_LOG_WARN(
          "[worker] still waiting for peer(s) to register: {} "
          "(resolved={} pending={}; readyz stays not-ready)",
          missing, resolved.size(), pending.size());
      lastWarn = now;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
  }

  if (!pending.empty()) {
    TT_LOG_WARN(
        "[worker] peer discovery cancelled with {} peer(s) still unresolved",
        pending.size());
    return std::nullopt;
  }
  TT_LOG_INFO("[worker] all {} peer(s) resolved", resolved.size());
  return resolved;
}

// Block until every created control channel is TCP-connected, or @p stop.
// Short await slices so SIGTERM is prompt; WARN heartbeat matches discovery.
bool awaitAllChannelsConnected(KvControlChannelConnector& connector,
                               const std::atomic<bool>& stop) {
  const std::size_t total = connector.channelCount();
  if (total == 0) return true;

  auto lastWarn = std::chrono::steady_clock::now();
  TT_LOG_INFO("[worker] waiting for {}/{} decode control channel(s) to connect",
              total, total);
  while (!stop.load()) {
    const std::size_t connected = connector.awaitConnected(
        std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
    if (connected >= total) {
      TT_LOG_INFO("[worker] all {} decode control channel(s) connected", total);
      return true;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - lastWarn >= std::chrono::milliseconds(K_DISCOVERY_TIMEOUT_MS)) {
      TT_LOG_WARN(
          "[worker] still waiting for decode control channels: {}/{} connected "
          "(readyz stays not-ready; not proceeding degraded)",
          connected, total);
      lastWarn = now;
    }
  }
  TT_LOG_WARN(
      "[worker] connect barrier cancelled with {}/{} channels connected",
      connector.connectedCount(), total);
  return false;
}

// TABLE_EXCHANGE with every connected decode: prefill keeps one fleet decode
// .pb (identical on all peers); each decode stores the prefill .pb. Block until
// all succeed or @p stop (readyz stays 503).
std::shared_ptr<const IKvTable> awaitDecodeTableFromControl(
    KvControlChannelConnector& connector,
    const std::vector<uint8_t>& prefillBlob, const std::atomic<bool>& stop) {
  auto lastWarn = std::chrono::steady_clock::now();
  TT_LOG_INFO(
      "[worker] waiting for control TABLE_EXCHANGE with all decode peers "
      "(local prefill blob {} B)",
      prefillBlob.size());

  while (!stop.load()) {
    connector.awaitConnected(std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
    if (connector.channelCount() == 0) {
      return nullptr;
    }
    if (connector.connectedCount() < connector.channelCount()) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
      continue;
    }

    std::shared_ptr<const IKvTable> decodeTable;
    bool allOk = true;
    for (const auto& [host, channel] : connector.channels()) {
      if (channel == nullptr || !channel->isConnected()) {
        allOk = false;
        break;
      }
      auto table =
          provisionPeerTable(*channel, TableExchangeRole::Sender, prefillBlob,
                             kDefaultTableExchangeTimeout);
      if (!table) {
        TT_LOG_WARN(
            "[worker] TABLE_EXCHANGE with '{}' failed; retrying all peers",
            host);
        allOk = false;
        break;
      }
      if (!decodeTable) {
        decodeTable = std::move(table);
      }
      TT_LOG_INFO(
          "[worker] TABLE_EXCHANGE with '{}' ok ({} B local prefill blob)",
          host, prefillBlob.size());
    }
    if (allOk && decodeTable) {
      return decodeTable;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - lastWarn >= std::chrono::milliseconds(K_DISCOVERY_TIMEOUT_MS)) {
      TT_LOG_WARN(
          "[worker] still waiting for control TABLE_EXCHANGE "
          "({}/{} channels connected; readyz stays not-ready)",
          connector.connectedCount(), connector.channelCount());
      lastWarn = now;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
  }

  TT_LOG_WARN(
      "[worker] TABLE_EXCHANGE cancelled before a decode table arrived");
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

  auto resolved =
      resolveWorkerTables(cfg, cfg.migrationMode == MigrationMode::DEVICE);
  if (!resolved) {
    if (gStop.load()) {
      TT_LOG_WARN(
          "[worker] prefill '{}' shutting down during engine table resolve",
          cfg.name);
      return 0;
    }
    TT_LOG_ERROR("[worker] failed to resolve prefill table + device map");
    return 1;
  }
  std::unique_ptr<MultiDeviceUmd> device;
  if (cfg.migrationMode == MigrationMode::DEVICE) {
    device = buildDeviceIo(*resolved->table, cfg.host, resolved->deviceMap);
    if (!device) {
      TT_LOG_ERROR("[worker] prefill '{}' failed to open devices (DeviceMap)",
                   cfg.name);
      return 1;
    }
  } else {
    TT_LOG_WARN(
        "[worker] prefill '{}' running in dry-run mode: device I/O and "
        "Mooncake data transfer are disabled",
        cfg.name);
  }

  // Full-mesh barrier (k8s-friendly): stay alive with /healthz up and /readyz
  // 503 until EVERY configured decode peer is resolved AND TCP-connected. Never
  // proceed degraded — migrate() needs the full host set anyway. SIGTERM aborts
  // the wait and exits without Ready.
  auto peersOpt = resolveAllPeers(*engine, cfg, gStop);
  if (!peersOpt) {
    TT_LOG_WARN("[worker] prefill '{}' shutting down during peer discovery",
                cfg.name);
    return 0;
  }
  auto& peers = *peersOpt;

  std::size_t configuredPeers = cfg.peers.size();
  for (const auto& name : cfg.discover_peers) {
    if (cfg.peers.count(name) == 0) ++configuredPeers;
  }

  KvControlChannelConnector connector(
      peers, makeClientTransport,
      std::chrono::milliseconds(K_MIGRATE_CONTROL_TIMEOUT_MS));
  if (!connector.openChannels() ||
      (configuredPeers > 0 && connector.channelCount() < configuredPeers)) {
    TT_LOG_ERROR(
        "[worker] failed to open control channels for all configured peers "
        "(opened={}/{})",
        connector.channelCount(), configuredPeers);
    return 1;
  }
  if (!awaitAllChannelsConnected(connector, gStop)) {
    TT_LOG_WARN("[worker] prefill '{}' shutting down during connect barrier",
                cfg.name);
    return 0;
  }

  // One fleet decode table: pull once from any connected decode, or disk
  // fallback when running without peers. Same barrier policy as discovery /
  // connect: stay alive with readyz 503 until exchange succeeds (or SIGTERM).
  std::shared_ptr<const IKvTable> decodeTable;
  const bool wantsExchange = configuredPeers > 0;
  if (wantsExchange) {
    decodeTable = awaitDecodeTableFromControl(connector, resolved->blob, gStop);
    if (!decodeTable) {
      TT_LOG_WARN("[worker] prefill '{}' shutting down during TABLE_EXCHANGE",
                  cfg.name);
      return 0;
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

  std::unique_ptr<KvMigrationMultiHostSender> sender;
  std::unique_ptr<tt::worker::IMigrationExecutor> executor;
  if (cfg.migrationMode == MigrationMode::DRY_RUN) {
    executor = std::make_unique<tt::worker::StubMigrationExecutor>();
  } else {
    sender = std::make_unique<KvMigrationMultiHostSender>(
        engine, *device, resolved->table, decodeTable, cfg.host,
        connector.channels(), &health);
    executor = std::make_unique<MooncakeMigrationExecutor>(*sender);
  }

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

  // Full mesh is up: Ready. Prefill /readyz stays latched (a peer outage must
  // not remove this worker from service — that peer's own probe handles it).
  // Mesh watch: re-resolve kv_control/<name> from metadata (same path as
  // bring-up). If the endpoint moved, replaceChannel + refresh the sender's
  // channel pointer; when TCP is up again, re-run TABLE_EXCHANGE and
  // refreshSegment so Mooncake WRITEs do not target a pre-restart address.
  // Same-host restarts can still heal sticky TCP without a metadata change;
  // the UP path still force-refreshes the data-plane segment.
  health.setLifecycle(WorkerLifecycle::Ready);
  TT_LOG_INFO(
      "[worker] prefill '{}' READY: {}/{} decode channels connected, "
      "mode={} brokers={} req={} ack={}",
      cfg.name, connector.connectedCount(), connector.channelCount(),
      cfg.migrationMode == MigrationMode::DRY_RUN ? "dry-run" : "device",
      brokers, reqTopic, ackTopic);
  worker.start();

  std::vector<std::string> peerNames;
  peerNames.reserve(peers.size());
  for (const auto& [name, _] : peers) {
    peerNames.push_back(name);
  }

  std::unordered_map<std::string, bool> wasConnected;
  for (const auto& name : peerNames) {
    const auto channels = connector.channels();
    const auto it = channels.find(name);
    wasConnected[name] = it != channels.end() && it->second != nullptr &&
                         it->second->isConnected();
  }
  auto lastMeshWarn = std::chrono::steady_clock::now();

  while (!gStop.load()) {
    for (const auto& name : peerNames) {
      KvControlChannelConnector::Endpoint ep;
      // Static --peer-control wins; otherwise re-read kv_control/<name> (and
      // rpc_meta fallback) every poll — same resolveOnePeer as bring-up.
      const auto currentEp = connector.endpoint(name);
      const bool peerResolved =
          (cfg.peers.count(name) != 0)
              ? (ep = cfg.peers.at(name), true)
              : resolveOnePeer(*engine, cfg, name, ep,
                               currentEp ? &*currentEp : nullptr);
      if (!peerResolved) {
        if (wasConnected[name]) {
          TT_LOG_WARN(
              "[worker] prefill '{}': decode peer '{}' LOST (control endpoint "
              "not in metadata; {}/{} connected)",
              cfg.name, name, connector.connectedCount(),
              connector.channelCount());
        }
        wasConnected[name] = false;
        continue;
      }

      if (!currentEp || *currentEp != ep) {
        TT_LOG_INFO(
            "[worker] prefill '{}': rediscovered peer '{}' control {}:{} "
            "(was {}) — replacing channel",
            cfg.name, name, ep.host, ep.port,
            currentEp
                ? (currentEp->host + ":" + std::to_string(currentEp->port))
                : std::string("none"));
        if (!connector.replaceChannel(name, ep)) {
          wasConnected[name] = false;
          continue;
        }
        const auto channels = connector.channels();
        const auto chIt = channels.find(name);
        if (chIt == channels.end() ||
            (sender != nullptr && !sender->addHost(name, chIt->second))) {
          wasConnected[name] = false;
          continue;
        }
        // New dial: wait until connected before TABLE_EXCHANGE.
        wasConnected[name] = false;
      }

      const auto channels = connector.channels();
      const auto chIt = channels.find(name);
      const std::shared_ptr<KvControlChannel> channel =
          (chIt != channels.end()) ? chIt->second : nullptr;
      const bool now = channel != nullptr && channel->isConnected();
      const bool before = wasConnected[name];

      if (before && !now) {
        TT_LOG_WARN(
            "[worker] prefill '{}': decode peer '{}' control channel LOST "
            "({}/{} still connected; rediscovering metadata)",
            cfg.name, name, connector.connectedCount(),
            connector.channelCount());
      } else if (!before && now) {
        TT_LOG_INFO(
            "[worker] prefill '{}': decode peer '{}' control channel UP "
            "({}/{}) — TABLE_EXCHANGE",
            cfg.name, name, connector.connectedCount(),
            connector.channelCount());
        // try_lock: if migrate() holds the channel transaction, skip and retry
        // next poll — never interleave TABLE_EXCHANGE with
        // Begin/Ready/Done/Ack.
        if (!tryProvisionPeerTable(*channel, TableExchangeRole::Sender,
                                   resolved->blob,
                                   kDefaultTableExchangeTimeout)) {
          TT_LOG_WARN(
              "[worker] TABLE_EXCHANGE with peer '{}' deferred or failed; "
              "will retry",
              name);
          continue;
        }
        TT_LOG_INFO("[worker] TABLE_EXCHANGE with peer '{}' succeeded", name);
        if (cfg.migrationMode == MigrationMode::DRY_RUN) {
          TT_LOG_INFO(
              "[worker] dry-run mode skips Mooncake segment refresh for '{}'",
              name);
          wasConnected[name] = now;
          continue;
        }
        if (engine->refreshSegment(name) == K_INVALID_SEGMENT) {
          TT_LOG_WARN(
              "[worker] refreshSegment('{}') after control UP failed; next "
              "migrate may still hit a stale data-plane address until a "
              "WRITE failure triggers reactive refresh",
              name);
        } else {
          TT_LOG_INFO(
              "[worker] refreshed Mooncake segment '{}' after control UP",
              name);
        }
      }
      wasConnected[name] = now;
    }

    const std::size_t connected = connector.connectedCount();
    const std::size_t total = connector.channelCount();
    if (total > 0 && connected < total) {
      const auto now = std::chrono::steady_clock::now();
      if (now - lastMeshWarn >=
          std::chrono::milliseconds(K_DISCOVERY_TIMEOUT_MS)) {
        TT_LOG_WARN(
            "[worker] prefill '{}': control mesh degraded {}/{} connected "
            "(readyz stays ready; waiting on metadata republish / TCP / "
            "TABLE_EXCHANGE)",
            cfg.name, connected, total);
        lastMeshWarn = now;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(K_DISCOVERY_POLL_MS));
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

  auto resolved =
      resolveWorkerTables(cfg, cfg.migrationMode == MigrationMode::DEVICE);
  if (!resolved) {
    if (gStop.load()) {
      TT_LOG_WARN(
          "[worker] decode '{}' shutting down during engine table resolve",
          cfg.name);
      return 0;
    }
    TT_LOG_ERROR("[worker] failed to resolve decode table + device map");
    return 1;
  }
  std::unique_ptr<MultiDeviceUmd> device;
  std::unique_ptr<MooncakeKvReceiver> receiver;
  std::string segment;
  if (cfg.migrationMode == MigrationMode::DEVICE) {
    device = buildDeviceIo(*resolved->table, cfg.host, resolved->deviceMap);
    if (!device) {
      TT_LOG_ERROR("[worker] decode '{}' failed to open devices (DeviceMap)",
                   cfg.name);
      return 1;
    }

    // Segment name the sender opens for the data plane. With a metadata service
    // the mirror is registered under — and resolvable by — the worker's LOGICAL
    // name (cfg.name): the same register-by-name / resolve-by-name discovery
    // the bringup worker uses on main (PeerDiscoveryService), so the sender
    // finds the peer through the metadata service instead of a hard-coded
    // endpoint. Only P2PHANDSHAKE (no metadata registry to resolve a logical
    // name) falls back to the engine's live IP:port. An explicit --segment
    // always overrides.
    segment = cfg.segment;
    if (segment.empty()) {
      segment = (cfg.metadata_uri == "P2PHANDSHAKE") ? engine->localServerName()
                                                     : cfg.name;
    }
    receiver = std::make_unique<MooncakeKvReceiver>(
        engine, *device, resolved->table, cfg.host, segment);
    if (!receiver->registered()) {
      TT_LOG_ERROR(
          "[worker] decode '{}' failed to register mirror segment '{}'",
          cfg.name, segment);
      return 1;
    }
  } else {
    TT_LOG_WARN(
        "[worker] decode '{}' running in dry-run mode: device I/O and "
        "Mooncake mirror registration are disabled",
        cfg.name);
  }

  // Control server stores peer prefill .pb on TABLE_EXCHANGE, replies with
  // this decode .pb, then serves migrate Begin/Done. Long receive timeout
  // covers large table provisioning.
  KvMigrationReceiverServer server{cfg.control_port, makeServerTransport,
                                   receiver.get(), resolved->blob,
                                   kDefaultTableExchangeTimeout};
  if (!server.start()) {
    TT_LOG_ERROR("[worker] decode '{}' failed to start control server on :{}",
                 cfg.name, cfg.control_port);
    return 1;
  }

  // Publish our KV control endpoint (host + bound port) into the metadata
  // service, keyed by our logical name, so prefill discovers where to open the
  // control channel — the same register-by-name / resolve-by-name path the data
  // plane already uses. The host mirrors what Mooncake advertised in rpc_meta.
  // Only meaningful with a real metadata service; under P2PHANDSHAKE this
  // no-ops and prefill falls back to a static endpoint.
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

  // Role-agnostic peer discovery is available for a future symmetric data
  // plane, but a decode is a pure receiver today: unused --peer entries must
  // NOT gate Ready. Prefill needs this decode's control port up; blocking on
  // resolveAllPeers here wedged rolling deploys when a listed peer was missing.
  if (!cfg.peers.empty() || !cfg.discover_peers.empty()) {
    TT_LOG_INFO(
        "[worker] decode '{}' ignoring {} configured peer(s) for readiness "
        "(receiver does not initiate migrations)",
        cfg.name, cfg.peers.size() + cfg.discover_peers.size());
  }

  // Control server listening and endpoint published: the decode worker has
  // finished its own bring-up, so flip it Ready.
  health.setLifecycle(WorkerLifecycle::Ready);
  TT_LOG_INFO(
      "[worker] decode '{}' READY: mode={} segment={} control_port={}",
      cfg.name,
      cfg.migrationMode == MigrationMode::DRY_RUN ? "dry-run" : "device",
      segment.empty() ? "disabled" : segment, cfg.control_port);
  while (!gStop.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(K_IDLE_POLL_MS));
  }
  server.stop();
  health.setLifecycle(WorkerLifecycle::ShuttingDown);
  // Drop our control endpoint before destructors so prefills stop resolving
  // this host immediately. rpc_meta is cleared by TransferEngine::freeEngine
  // when `engine`/`receiver` shared_ptrs die — SIGKILL still needs deploy/
  // watchdog clearRpcMeta.
  if (engine->removeMetadata(controlKey)) {
    TT_LOG_INFO("[worker] decode '{}' cleared control endpoint {}", cfg.name,
                controlKey);
  }
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
