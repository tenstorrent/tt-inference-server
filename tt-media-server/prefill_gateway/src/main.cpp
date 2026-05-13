// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/dispatcher.hpp"
#include "gateway/prefill_registry.hpp"
#include "sockets/socket_manager.hpp"
#include "sockets/socket_messages.hpp"
#include "utils/logger.hpp"

// ── Config ────────────────────────────────────────────────────────────────────

namespace {

struct PrefillEndpoint {
  std::string host;
  uint16_t port;
};

struct GatewayConfig {
  uint16_t decode_port = 0;
  std::vector<PrefillEndpoint> prefills;
};

void printUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " --decode-port=<PORT> --prefill=<HOST>:<PORT> "
      << "[--prefill=<HOST>:<PORT> ...]\n\n"
      << "  --decode-port=PORT   Port the gateway listens on for decode.\n"
      << "  --prefill=HOST:PORT  Prefill server to connect to (repeatable).\n"
      << "  --help               Print this message.\n\n"
      << "Example:\n"
      << "  " << prog
      << " --decode-port=7100 --prefill=192.168.1.1:7200 --prefill=192.168.1.2:7200\n";
}

std::optional<PrefillEndpoint> parsePrefillArg(const std::string& value) {
  auto colon = value.rfind(':');
  if (colon == std::string::npos || colon == 0 ||
      colon == value.size() - 1) {
    return std::nullopt;
  }
  std::string host = value.substr(0, colon);
  int port_int = std::stoi(value.substr(colon + 1));
  if (port_int <= 0 || port_int > 65535) return std::nullopt;
  return PrefillEndpoint{std::move(host), static_cast<uint16_t>(port_int)};
}

std::optional<std::string_view> flagValue(std::string_view arg,
                                          std::string_view prefix) {
  if (arg.substr(0, prefix.size()) == prefix) {
    return arg.substr(prefix.size());
  }
  return std::nullopt;
}

std::optional<GatewayConfig> parseArgs(int argc, char** argv) {
  GatewayConfig cfg;

  for (int i = 1; i < argc; ++i) {
    std::string_view arg(argv[i]);

    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return std::nullopt;
    }

    if (auto v = flagValue(arg, "--decode-port=")) {
      cfg.decode_port = static_cast<uint16_t>(std::stoi(std::string(*v)));
      continue;
    }

    if (auto v = flagValue(arg, "--prefill=")) {
      auto ep = parsePrefillArg(std::string(*v));
      if (!ep) {
        std::cerr << "Invalid --prefill value: " << *v
                  << " (expected HOST:PORT)\n";
        return std::nullopt;
      }
      cfg.prefills.push_back(std::move(*ep));
      continue;
    }

    std::cerr << "Unknown argument: " << arg << "\n";
    printUsage(argv[0]);
    return std::nullopt;
  }

  if (cfg.decode_port == 0) {
    std::cerr << "--decode-port is required.\n";
    printUsage(argv[0]);
    return std::nullopt;
  }

  if (cfg.prefills.empty()) {
    std::cerr << "At least one --prefill is required.\n";
    printUsage(argv[0]);
    return std::nullopt;
  }

  return cfg;
}

// ── Signal handling ───────────────────────────────────────────────────────────

volatile sig_atomic_t g_stop = 0;

void signalHandler(int /*sig*/) { g_stop = 1; }

}  // namespace

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize();

  auto cfg_opt = parseArgs(argc, argv);
  if (!cfg_opt) return 1;
  const GatewayConfig& cfg = *cfg_opt;

  TT_LOG_INFO("[Gateway] Starting — decode port={}, prefills={}",
              cfg.decode_port, cfg.prefills.size());

  // ── Core gateway objects ─────────────────────────────────────────────────
  tt::gateway::PrefillRegistry registry;
  tt::gateway::AffinityCache affinity;

  // ── Decode-facing SocketManager (SERVER mode) ─────────────────────────
  // The gateway listens on decode_port; the single decode server connects
  // here to submit PrefillRequests and receive PrefillResults /
  // PrefillAssignments. Only one concurrent decode connection is needed
  // (1 decode : N prefills); the existing SERVER mode loop already handles
  // sequential reconnects automatically.
  tt::sockets::SocketManager decode_sm;
  if (!decode_sm.initializeAsServer(cfg.decode_port)) {
    TT_LOG_ERROR("[Gateway] Failed to bind decode port {}", cfg.decode_port);
    return 1;
  }

  // ── Per-prefill SocketManagers (CLIENT mode, one per endpoint) ────────
  // The gateway connects outward to each prefill. Using N independent
  // SocketManager instances is the gateway's implementation of #3508
  // (1:N multi-peer): each instance owns its own socket, reconnect loop,
  // send mutex, and message thread. This avoids changing the existing 1:1
  // SocketManager contract while giving the gateway full independence per peer.
  std::vector<std::unique_ptr<tt::sockets::SocketManager>> prefill_sms;
  prefill_sms.reserve(cfg.prefills.size());
  for (const auto& ep : cfg.prefills) {
    auto sm = std::make_unique<tt::sockets::SocketManager>();
    // Exponential backoff: 100ms → 200ms → … → 5s (#3509)
    sm->setReconnectBackoff(/*initial_ms=*/100, /*max_ms=*/5000);
    if (!sm->initializeAsClient(ep.host, ep.port)) {
      TT_LOG_ERROR("[Gateway] Failed to init client socket to {}:{}", ep.host,
                   ep.port);
      return 1;
    }
    prefill_sms.push_back(std::move(sm));
  }

  // ── Dispatcher: built with injectable Senders so it stays socket-free ─
  // We forward-declare the pointer here; the Senders below capture it by
  // pointer so they remain valid after the unique_ptr is assigned below.
  std::unique_ptr<tt::gateway::Dispatcher> dispatcher_ptr;

  tt::gateway::Dispatcher::Senders senders;

  senders.sendRequestToPrefill =
      [&registry](const std::string& server_id,
                  const tt::sockets::PrefillRequestMessage& msg) -> bool {
    auto* sm = registry.getSocketManager(server_id);
    if (!sm) {
      TT_LOG_WARN("[Gateway] sendRequestToPrefill: no socket for '{}'",
                  server_id);
      return false;
    }
    return sm->sendObject("prefill_request", msg);
  };

  senders.sendAssignmentToDecode =
      [&decode_sm](const tt::sockets::PrefillAssignmentMessage& msg) -> bool {
    return decode_sm.sendObject(tt::sockets::tags::PREFILL_ASSIGNMENT, msg);
  };

  senders.sendResultToDecode =
      [&decode_sm](const tt::sockets::PrefillResultMessage& msg) -> bool {
    return decode_sm.sendObject("prefill_result", msg);
  };

  dispatcher_ptr = std::make_unique<tt::gateway::Dispatcher>(
      registry, affinity, std::move(senders));

  // Wire prefill-down callback: registry notifies dispatcher so in-flight
  // tasks are failed back to decode when a prefill disappears.
  registry.setOnPrefillDown([&dispatcher_ptr](const std::string& id) {
    dispatcher_ptr->onPrefillDown(id);
  });

  // ── Wire handlers for each prefill SocketManager ─────────────────────
  for (auto& sm_ptr : prefill_sms) {
    tt::sockets::SocketManager* sm = sm_ptr.get();

    // server_id_holder is shared between the registration handler (which sets
    // it on first connect) and the connection-lost callback (which reads it).
    // Using a shared_ptr<string> avoids dangling references if the lambda
    // outlives the stack frame, and allows the lost-callback to see the id
    // that was learned from the registration message.
    auto id_holder = std::make_shared<std::string>();

    sm->registerHandler<tt::sockets::PrefillRegistrationMessage>(
        tt::sockets::tags::PREFILL_REGISTRATION,
        [&registry, sm, id_holder](
            const tt::sockets::PrefillRegistrationMessage& msg) {
          TT_LOG_INFO("[Gateway] Prefill registered: id='{}' max_in_flight={}",
                      msg.server_id, msg.max_in_flight);
          *id_holder = msg.server_id;
          // preRegister is a no-op if the entry already exists (reconnect).
          registry.preRegister(msg.server_id, sm);
          bool ok = registry.markRegistered(msg.server_id, msg.max_in_flight);
          if (!ok) {
            TT_LOG_ERROR(
                "[Gateway] markRegistered failed for '{}' — this is a bug",
                msg.server_id);
          }
        });

    sm->registerHandler<tt::sockets::PrefillResultMessage>(
        "prefill_result",
        [&dispatcher_ptr, id_holder](
            const tt::sockets::PrefillResultMessage& msg) {
          dispatcher_ptr->onPrefillResult(*id_holder, msg);
        });

    sm->registerHandler<tt::sockets::PrefillCacheBlocksAddedMessage>(
        tt::sockets::tags::PREFILL_CACHE_BLOCKS_ADDED,
        [&dispatcher_ptr](
            const tt::sockets::PrefillCacheBlocksAddedMessage& msg) {
          dispatcher_ptr->onCacheBlocksAdded(msg);
        });

    sm->registerHandler<tt::sockets::PrefillCacheBlocksEvictedMessage>(
        tt::sockets::tags::PREFILL_CACHE_BLOCKS_EVICTED,
        [&dispatcher_ptr](
            const tt::sockets::PrefillCacheBlocksEvictedMessage& msg) {
          dispatcher_ptr->onCacheBlocksEvicted(msg);
        });

    sm->setConnectionLostCallback([&registry, id_holder]() {
      const std::string& sid = *id_holder;
      if (!sid.empty()) {
        TT_LOG_WARN("[Gateway] Prefill '{}' connection lost", sid);
        registry.markDown(sid);
        // dispatcher_.onPrefillDown is triggered via the registry callback set
        // above, so we do not call it directly here.
      }
    });
  }

  // ── Wire decode SocketManager ─────────────────────────────────────────
  decode_sm.registerHandler<tt::sockets::PrefillRequestMessage>(
      "prefill_request",
      [&dispatcher_ptr](const tt::sockets::PrefillRequestMessage& msg) {
        dispatcher_ptr->onPrefillRequest(msg);
      });

  decode_sm.setConnectionLostCallback([]() {
    TT_LOG_WARN("[Gateway] Decode disconnected — waiting for reconnect");
  });

  // ── Start ────────────────────────────────────────────────────────────
  for (auto& sm : prefill_sms) sm->start();
  decode_sm.start();

  TT_LOG_INFO("[Gateway] Running. Send SIGINT/SIGTERM to stop.");

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  while (!g_stop) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  // ── Graceful shutdown ────────────────────────────────────────────────
  TT_LOG_INFO("[Gateway] Shutting down…");
  decode_sm.stop();
  for (auto& sm : prefill_sms) sm->stop();
  TT_LOG_INFO("[Gateway] Stopped.");

  return 0;
}
