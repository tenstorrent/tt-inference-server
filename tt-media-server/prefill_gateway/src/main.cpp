// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/dispatcher.hpp"
#include "gateway/prefill_registry.hpp"
#include "sockets/socket_manager.hpp"
#include "sockets/socket_messages.hpp"
#include "utils/logger.hpp"

namespace {

struct PrefillEndpoint {
  std::string host;
  uint16_t port;
};

struct GatewayConfig {
  uint16_t decodePort = 0;
  std::vector<PrefillEndpoint> prefills;
};

struct PrefillConnectionState {
  void setServerId(const std::string& serverId) {
    std::lock_guard<std::mutex> lock(mutex);
    this->serverId = serverId;
  }

  std::string getServerId() const {
    std::lock_guard<std::mutex> lock(mutex);
    return serverId;
  }

  mutable std::mutex mutex;
  std::string serverId;
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
      << " --decode-port=7100 --prefill=192.168.1.1:7200 "
         "--prefill=192.168.1.2:7200\n";
}

std::optional<PrefillEndpoint> parsePrefillArg(const std::string& value) {
  auto colon = value.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon == value.size() - 1) {
    return std::nullopt;
  }
  std::string host = value.substr(0, colon);
  int portInt = std::stoi(value.substr(colon + 1));
  if (portInt <= 0 || portInt > 65535) return std::nullopt;
  return PrefillEndpoint{std::move(host), static_cast<uint16_t>(portInt)};
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
      cfg.decodePort = static_cast<uint16_t>(std::stoi(std::string(*v)));
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

  if (cfg.decodePort == 0) {
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

volatile sig_atomic_t gStop = 0;

void signalHandler(int /*sig*/) { gStop = 1; }

}  // namespace

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize();

  auto cfgOpt = parseArgs(argc, argv);
  if (!cfgOpt) return 1;
  const GatewayConfig& cfg = *cfgOpt;

  TT_LOG_INFO("[Gateway] Starting — decode port={}, prefills={}",
              cfg.decodePort, cfg.prefills.size());

  tt::gateway::PrefillRegistry registry;
  tt::gateway::AffinityCache affinity;

  // Decode-facing: gateway listens, decode dials in (only 1 decode connection).
  tt::sockets::SocketManager decodeSm;
  if (!decodeSm.initializeAsServer(cfg.decodePort)) {
    TT_LOG_ERROR("[Gateway] Failed to bind decode port {}", cfg.decodePort);
    return 1;
  }

  // Per-prefill: one independent SocketManager (CLIENT) per endpoint — the
  // gateway's 1:N fan-out without modifying the underlying 1:1 transport.
  std::vector<std::unique_ptr<tt::sockets::SocketManager>> prefillSms;
  prefillSms.reserve(cfg.prefills.size());
  for (const auto& ep : cfg.prefills) {
    auto sm = std::make_unique<tt::sockets::SocketManager>();
    sm->setReconnectBackoff(/*initial_ms=*/1000, /*max_ms=*/5000);
    if (!sm->initializeAsClient(ep.host, ep.port)) {
      TT_LOG_ERROR("[Gateway] Failed to init client socket to {}:{}", ep.host,
                   ep.port);
      return 1;
    }
    prefillSms.push_back(std::move(sm));
  }

  // Dispatcher takes Senders by value; lambdas below capture references which
  // stay alive for the rest of main.
  std::unique_ptr<tt::gateway::Dispatcher> dispatcherPtr;

  tt::gateway::Dispatcher::Senders senders;

  senders.sendRequestToPrefill =
      [&registry](const std::string& serverId,
                  const tt::sockets::PrefillRequestMessage& msg) -> bool {
    auto* sm = registry.getSocketManager(serverId);
    if (!sm) {
      TT_LOG_WARN("[Gateway] sendRequestToPrefill: no socket for '{}'",
                  serverId);
      return false;
    }
    return sm->sendObject("prefill_request", msg);
  };

  senders.sendAssignmentToDecode =
      [&decodeSm](const tt::sockets::PrefillAssignmentMessage& msg) -> bool {
    return decodeSm.sendObject(tt::sockets::tags::PREFILL_ASSIGNMENT, msg);
  };

  senders.sendResultToDecode =
      [&decodeSm](const tt::sockets::PrefillResultMessage& msg) -> bool {
    return decodeSm.sendObject("prefill_result", msg);
  };

  dispatcherPtr = std::make_unique<tt::gateway::Dispatcher>(registry, affinity,
                                                            std::move(senders));

  registry.setOnPrefillDown([&dispatcherPtr](const std::string& id) {
    dispatcherPtr->onPrefillDown(id);
  });

  for (auto& smPtr : prefillSms) {
    tt::sockets::SocketManager* sm = smPtr.get();

    // Shared between callbacks that may run on different threads. The id is
    // unknown until the first registration message.
    auto state = std::make_shared<PrefillConnectionState>();

    sm->registerHandler<tt::sockets::PrefillRegistrationMessage>(
        tt::sockets::tags::PREFILL_REGISTRATION,
        [&registry, sm,
         state](const tt::sockets::PrefillRegistrationMessage& msg) {
          TT_LOG_DEBUG("[Gateway] Prefill registered: id='{}' max_in_flight={}",
                       msg.server_id, msg.max_in_flight);
          state->setServerId(msg.server_id);
          registry.preRegister(msg.server_id, sm);
          bool ok = registry.markRegistered(msg.server_id, msg.max_in_flight);
          if (!ok) {
            TT_LOG_ERROR("[Gateway] markRegistered failed for '{}'",
                         msg.server_id);
          }
        });

    sm->registerHandler<tt::sockets::PrefillResultMessage>(
        "prefill_result", [&dispatcherPtr, state](
                              const tt::sockets::PrefillResultMessage& msg) {
          dispatcherPtr->onPrefillResult(state->getServerId(), msg);
        });

    sm->registerHandler<tt::sockets::PrefillCacheBlocksAddedMessage>(
        tt::sockets::tags::PREFILL_CACHE_BLOCKS_ADDED,
        [&dispatcherPtr](
            const tt::sockets::PrefillCacheBlocksAddedMessage& msg) {
          dispatcherPtr->onCacheBlocksAdded(msg);
        });

    sm->registerHandler<tt::sockets::PrefillCacheBlocksEvictedMessage>(
        tt::sockets::tags::PREFILL_CACHE_BLOCKS_EVICTED,
        [&dispatcherPtr](
            const tt::sockets::PrefillCacheBlocksEvictedMessage& msg) {
          dispatcherPtr->onCacheBlocksEvicted(msg);
        });

    sm->setConnectionLostCallback([&registry, state]() {
      const std::string sid = state->getServerId();
      if (!sid.empty()) {
        TT_LOG_WARN("[Gateway] Prefill '{}' connection lost", sid);
        registry.markDown(sid);  // fires onPrefillDown -> dispatcher
      }
    });
  }

  decodeSm.registerHandler<tt::sockets::PrefillRequestMessage>(
      "prefill_request",
      [&dispatcherPtr](const tt::sockets::PrefillRequestMessage& msg) {
        dispatcherPtr->onPrefillRequest(msg);
      });

  decodeSm.setConnectionLostCallback([]() {
    TT_LOG_WARN("[Gateway] Decode disconnected — waiting for reconnect");
  });

  for (auto& sm : prefillSms) sm->start();
  decodeSm.start();

  std::atomic<bool> proberStop{false};
  constexpr auto probeIntervalMs = std::chrono::milliseconds(1000);
  std::thread proberThread([&prefillSms, &proberStop, probeIntervalMs]() {
    while (!proberStop.load()) {
      for (auto& sm : prefillSms) {
        sm->sendObject(tt::sockets::tags::REGISTRATION_PROBE,
                       tt::sockets::RegistrationProbeMessage{});
      }
      std::this_thread::sleep_for(probeIntervalMs);
    }
  });

  TT_LOG_INFO("[Gateway] Running. Send SIGINT/SIGTERM to stop.");

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  while (!gStop) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  TT_LOG_INFO("[Gateway] Shutting down…");
  proberStop = true;
  if (proberThread.joinable()) proberThread.join();
  decodeSm.stop();
  for (auto& sm : prefillSms) sm->stop();
  TT_LOG_INFO("[Gateway] Stopped.");

  return 0;
}
