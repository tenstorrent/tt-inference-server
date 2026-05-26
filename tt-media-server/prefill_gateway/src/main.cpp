// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/dispatcher.hpp"
#include "gateway/prefill_connection_wiring.hpp"
#include "gateway/prefill_registry.hpp"
#include "gateway/zmq_prefill_router.hpp"
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
  std::string prefillBindHost = "*";
  uint16_t prefillBindPort = 0;
  uint32_t prefillStaleTimeoutMs = 3000;
  uint32_t requestTimeoutMs = 300000;
  uint32_t timeoutWindowMs = 60000;
  uint32_t timeoutCooldownMs = 30000;
  uint32_t timeoutThreshold = 3;
};

void printUsage(const char* prog) {
  std::cerr
      << "Usage: " << prog << " --decode-port=<PORT> --prefill=<HOST>:<PORT> "
      << "[--prefill=<HOST>:<PORT> ...]\n\n"
      << "  --decode-port=PORT   Port the gateway listens on for decode.\n"
      << "  --prefill=HOST:PORT  TCP prefill server to connect to "
         "(repeatable).\n"
      << "  --prefill-bind=HOST:PORT\n"
      << "                        ZMQ ROUTER bind endpoint for prefills.\n"
      << "  --prefill-stale-timeout-ms=MS\n"
      << "                        ZMQ prefill registration timeout. Default: "
         "3000.\n"
      << "  --request-timeout-ms=MS\n"
      << "                        In-flight prefill request timeout. Use 0 to "
         "disable. Default: 300000.\n"
      << "  --timeout-window-ms=MS\n"
      << "                        Window for repeated timeout detection. "
         "Default: "
         "60000.\n"
      << "  --timeout-cooldown-ms=MS\n"
      << "                        Time to stop routing new tasks to a prefill "
         "after repeated timeouts. Default: 30000.\n"
      << "  --timeout-threshold=N\n"
      << "                        Timeouts in the window before cooldown. Use "
         "0 to "
         "disable. Default: 3.\n"
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

    if (auto v = flagValue(arg, "--prefill-stale-timeout-ms=")) {
      int timeoutMs = std::stoi(std::string(*v));
      if (timeoutMs <= 0) {
        std::cerr << "Invalid --prefill-stale-timeout-ms value: " << *v
                  << " (expected positive milliseconds)\n";
        return std::nullopt;
      }
      cfg.prefillStaleTimeoutMs = static_cast<uint32_t>(timeoutMs);
      continue;
    }

    if (auto v = flagValue(arg, "--request-timeout-ms=")) {
      int timeoutMs = std::stoi(std::string(*v));
      if (timeoutMs < 0) {
        std::cerr << "Invalid --request-timeout-ms value: " << *v
                  << " (expected non-negative milliseconds)\n";
        return std::nullopt;
      }
      cfg.requestTimeoutMs = static_cast<uint32_t>(timeoutMs);
      continue;
    }

    if (auto v = flagValue(arg, "--timeout-window-ms=")) {
      int timeoutMs = std::stoi(std::string(*v));
      if (timeoutMs < 0) {
        std::cerr << "Invalid --timeout-window-ms value: " << *v
                  << " (expected non-negative milliseconds)\n";
        return std::nullopt;
      }
      cfg.timeoutWindowMs = static_cast<uint32_t>(timeoutMs);
      continue;
    }

    if (auto v = flagValue(arg, "--timeout-cooldown-ms=")) {
      int timeoutMs = std::stoi(std::string(*v));
      if (timeoutMs < 0) {
        std::cerr << "Invalid --timeout-cooldown-ms value: " << *v
                  << " (expected non-negative milliseconds)\n";
        return std::nullopt;
      }
      cfg.timeoutCooldownMs = static_cast<uint32_t>(timeoutMs);
      continue;
    }

    if (auto v = flagValue(arg, "--timeout-threshold=")) {
      int threshold = std::stoi(std::string(*v));
      if (threshold < 0) {
        std::cerr << "Invalid --timeout-threshold value: " << *v
                  << " (expected non-negative count)\n";
        return std::nullopt;
      }
      cfg.timeoutThreshold = static_cast<uint32_t>(threshold);
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

    if (auto v = flagValue(arg, "--prefill-bind=")) {
      auto ep = parsePrefillArg(std::string(*v));
      if (!ep) {
        std::cerr << "Invalid --prefill-bind value: " << *v
                  << " (expected HOST:PORT)\n";
        return std::nullopt;
      }
      cfg.prefillBindHost = std::move(ep->host);
      cfg.prefillBindPort = ep->port;
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

  return cfg;
}

std::string socketTransportFromEnv() {
  const char* value = std::getenv("SOCKET_TRANSPORT");
  return value ? std::string(value) : tt::sockets::transport_names::TCP;
}

volatile sig_atomic_t gStop = 0;

void signalHandler(int /*sig*/) { gStop = 1; }

}  // namespace

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize();

  auto cfgOpt = parseArgs(argc, argv);
  if (!cfgOpt) return 1;
  const GatewayConfig& cfg = *cfgOpt;
  const bool useZmqPrefillRouter =
      socketTransportFromEnv() == tt::sockets::transport_names::ZMQ;

  if (useZmqPrefillRouter && cfg.prefillBindPort == 0) {
    std::cerr
        << "--prefill-bind=HOST:PORT is required for SOCKET_TRANSPORT=zmq\n";
    return 1;
  }
  if (!useZmqPrefillRouter && cfg.prefills.empty()) {
    std::cerr << "At least one --prefill is required for TCP mode.\n";
    return 1;
  }
  if (useZmqPrefillRouter && !cfg.prefills.empty()) {
    TT_LOG_WARN("[Gateway] Ignoring --prefill endpoints in ZMQ ROUTER mode");
  }

  TT_LOG_INFO("[Gateway] Starting — decode port={}, transport={}",
              cfg.decodePort, useZmqPrefillRouter ? "zmq" : "tcp");

  tt::gateway::PrefillRegistry registry;
  tt::gateway::AffinityCache affinity;

  // Decode-facing: gateway listens, decode dials in (only 1 decode connection).
  tt::sockets::SocketManager decodeSm;
  if (!decodeSm.initializeAsServer(cfg.decodePort)) {
    TT_LOG_ERROR("[Gateway] Failed to bind decode port {}", cfg.decodePort);
    return 1;
  }

  tt::gateway::ZmqPrefillRouter zmqPrefillRouter;

  if (useZmqPrefillRouter &&
      !zmqPrefillRouter.start(cfg.prefillBindHost, cfg.prefillBindPort)) {
    TT_LOG_ERROR("[Gateway] Failed to bind ZMQ prefill ROUTER on {}:{}",
                 cfg.prefillBindHost, cfg.prefillBindPort);
    return 1;
  }

  // TCP keeps one independent SocketManager (CLIENT) per endpoint.
  tt::gateway::PrefillSocketManagers prefillSms;
  prefillSms.reserve(cfg.prefills.size());
  if (!useZmqPrefillRouter) {
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
  }

  // Dispatcher takes Senders by value; lambdas below capture references which
  // stay alive for the rest of main.
  std::unique_ptr<tt::gateway::Dispatcher> dispatcherPtr;

  tt::gateway::Dispatcher::Senders senders;

  senders.sendRequestToPrefill =
      [&registry, &zmqPrefillRouter, useZmqPrefillRouter](
          const std::string& serverId,
          const tt::sockets::PrefillRequestMessage& msg) -> bool {
    if (useZmqPrefillRouter) {
      return zmqPrefillRouter.sendObject(serverId, "prefill_request", msg);
    }

    auto* sm = registry.getSocketManager(serverId);
    if (!sm) {
      TT_LOG_WARN("[Gateway] sendRequestToPrefill: no socket for '{}'",
                  serverId);
      return false;
    }
    return sm->sendObject("prefill_request", msg);
  };

  senders.sendCancelToPrefill =
      [&registry, &zmqPrefillRouter, useZmqPrefillRouter](
          const std::string& serverId,
          const tt::sockets::CancelPrefillMessage& msg) -> bool {
    if (useZmqPrefillRouter) {
      return zmqPrefillRouter.sendObject(
          serverId, tt::sockets::tags::CANCEL_PREFILL, msg);
    }

    auto* sm = registry.getSocketManager(serverId);
    if (!sm) {
      TT_LOG_WARN("[Gateway] sendCancelToPrefill: no socket for '{}'",
                  serverId);
      return false;
    }
    return sm->sendObject(tt::sockets::tags::CANCEL_PREFILL, msg);
  };

  senders.sendAssignmentToDecode =
      [&decodeSm](const tt::sockets::PrefillAssignmentMessage& msg) -> bool {
    return decodeSm.sendObject(tt::sockets::tags::PREFILL_ASSIGNMENT, msg);
  };

  senders.sendResultToDecode =
      [&decodeSm](const tt::sockets::PrefillResultMessage& msg) -> bool {
    return decodeSm.sendObject("prefill_result", msg);
  };

  tt::gateway::Dispatcher::Options dispatcherOptions{
      std::chrono::milliseconds(cfg.requestTimeoutMs),
      std::chrono::milliseconds(cfg.timeoutWindowMs),
      std::chrono::milliseconds(cfg.timeoutCooldownMs), cfg.timeoutThreshold};
  dispatcherPtr = std::make_unique<tt::gateway::Dispatcher>(
      registry, affinity, std::move(senders), dispatcherOptions);

  registry.setOnPrefillDown([&dispatcherPtr](const std::string& id) {
    dispatcherPtr->onPrefillDown(id);
  });

  if (useZmqPrefillRouter) {
    tt::gateway::registerZmqPrefillHandlers(zmqPrefillRouter, registry,
                                            *dispatcherPtr);
  } else {
    tt::gateway::registerTcpPrefillHandlers(prefillSms, registry,
                                            *dispatcherPtr);
  }

  decodeSm.registerHandler<tt::sockets::PrefillRequestMessage>(
      "prefill_request",
      [&dispatcherPtr](const tt::sockets::PrefillRequestMessage& msg) {
        dispatcherPtr->onPrefillRequest(msg);
      });

  decodeSm.registerHandler<tt::sockets::CancelPrefillMessage>(
      tt::sockets::tags::CANCEL_PREFILL,
      [&dispatcherPtr](const tt::sockets::CancelPrefillMessage& msg) {
        dispatcherPtr->onPrefillCancel(msg);
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

  std::atomic<bool> watchdogStop{false};
  const auto prefillStaleTimeout =
      std::chrono::milliseconds(cfg.prefillStaleTimeoutMs);
  std::thread watchdogThread;
  if (useZmqPrefillRouter) {
    watchdogThread = std::thread(
        [&registry, &zmqPrefillRouter, &watchdogStop, prefillStaleTimeout] {
          while (!watchdogStop.load()) {
            for (const auto& serverId :
                 zmqPrefillRouter.takeStaleServers(prefillStaleTimeout)) {
              TT_LOG_WARN("[Gateway] Prefill '{}' registration timed out",
                          serverId);
              registry.markDown(serverId);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
          }
        });
  }

  std::atomic<bool> requestTimeoutStop{false};
  std::thread requestTimeoutThread;
  if (cfg.requestTimeoutMs > 0) {
    requestTimeoutThread = std::thread([&dispatcherPtr, &requestTimeoutStop] {
      while (!requestTimeoutStop.load()) {
        dispatcherPtr->onRequestTimeouts();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    });
  }

  TT_LOG_INFO("[Gateway] Running. Send SIGINT/SIGTERM to stop.");

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  while (!gStop) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  TT_LOG_INFO("[Gateway] Shutting down…");
  proberStop = true;
  if (proberThread.joinable()) proberThread.join();
  requestTimeoutStop = true;
  if (requestTimeoutThread.joinable()) requestTimeoutThread.join();
  watchdogStop = true;
  if (watchdogThread.joinable()) watchdogThread.join();
  decodeSm.stop();
  for (auto& sm : prefillSms) sm->stop();
  zmqPrefillRouter.stop();
  TT_LOG_INFO("[Gateway] Stopped.");

  return 0;
}
