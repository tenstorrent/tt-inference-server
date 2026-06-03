// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/dispatcher.hpp"
#include "gateway/gateway_health.hpp"
#include "gateway/gateway_health_server.hpp"
#include "gateway/gateway_metrics.hpp"
#include "gateway/gateway_metrics_server.hpp"
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
  std::chrono::milliseconds prefillStaleTimeout{3000};
  std::chrono::milliseconds requestTimeout{300000};
  std::chrono::milliseconds timeoutWindow{60000};
  std::chrono::milliseconds timeoutCooldown{30000};
  uint32_t timeoutThreshold = 3;
  uint16_t metricsPort = 9091;
  uint16_t healthPort = 0;
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
      << "  --metrics-port=PORT  Port for Prometheus GET /metrics. Use 0 to "
         "disable. Default: 9091.\n"
      << "  --health-port=PORT   Port for GET /tt-liveness and /health. Use "
         "0 to disable. Default: 0.\n"
      << "  --help               Print this message.\n\n"
      << "Example:\n"
      << "  " << prog
      << " --decode-port=7100 --prefill=192.168.1.1:7200 "
         "--prefill=192.168.1.2:7200\n";
}

std::optional<PrefillEndpoint> parsePrefillArg(std::string_view value) {
  auto colon = value.rfind(':');
  if (colon == std::string::npos || colon == 0 || colon == value.size() - 1) {
    return std::nullopt;
  }
  std::string host(value.substr(0, colon));
  int portInt = std::stoi(std::string(value.substr(colon + 1)));
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

std::optional<std::chrono::milliseconds> parseMilliseconds(
    std::string_view value, std::string_view flagName, bool allowZero) {
  const int timeoutMs = std::stoi(std::string(value));
  if (timeoutMs < 0 || (!allowZero && timeoutMs == 0)) {
    std::cerr << "Invalid " << flagName << " value: " << value << " (expected "
              << (allowZero ? "non-negative" : "positive")
              << " milliseconds)\n";
    return std::nullopt;
  }
  return std::chrono::milliseconds(timeoutMs);
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
      auto timeout = parseMilliseconds(*v, "--prefill-stale-timeout-ms",
                                       /*allowZero=*/false);
      if (!timeout) {
        return std::nullopt;
      }
      cfg.prefillStaleTimeout = *timeout;
      continue;
    }

    if (auto v = flagValue(arg, "--request-timeout-ms=")) {
      auto timeout =
          parseMilliseconds(*v, "--request-timeout-ms", /*allowZero=*/true);
      if (!timeout) {
        return std::nullopt;
      }
      cfg.requestTimeout = *timeout;
      continue;
    }

    if (auto v = flagValue(arg, "--timeout-window-ms=")) {
      auto timeout =
          parseMilliseconds(*v, "--timeout-window-ms", /*allowZero=*/true);
      if (!timeout) {
        return std::nullopt;
      }
      cfg.timeoutWindow = *timeout;
      continue;
    }

    if (auto v = flagValue(arg, "--timeout-cooldown-ms=")) {
      auto timeout =
          parseMilliseconds(*v, "--timeout-cooldown-ms", /*allowZero=*/true);
      if (!timeout) {
        return std::nullopt;
      }
      cfg.timeoutCooldown = *timeout;
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

    if (auto v = flagValue(arg, "--metrics-port=")) {
      int port = std::stoi(std::string(*v));
      if (port < 0 || port > 65535) {
        std::cerr << "Invalid --metrics-port value: " << *v
                  << " (expected 0-65535)\n";
        return std::nullopt;
      }
      cfg.metricsPort = static_cast<uint16_t>(port);
      continue;
    }

    if (auto v = flagValue(arg, "--health-port=")) {
      int port = std::stoi(std::string(*v));
      if (port < 0 || port > 65535) {
        std::cerr << "Invalid --health-port value: " << *v
                  << " (expected 0-65535)\n";
        return std::nullopt;
      }
      cfg.healthPort = static_cast<uint16_t>(port);
      continue;
    }

    if (auto v = flagValue(arg, "--prefill=")) {
      auto ep = parsePrefillArg(*v);
      if (!ep) {
        std::cerr << "Invalid --prefill value: " << *v
                  << " (expected HOST:PORT)\n";
        return std::nullopt;
      }
      cfg.prefills.push_back(std::move(*ep));
      continue;
    }

    if (auto v = flagValue(arg, "--prefill-bind=")) {
      auto ep = parsePrefillArg(*v);
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

std::string_view socketTransportFromEnv() {
  const char* value = std::getenv("SOCKET_TRANSPORT");
  return value ? std::string_view(value) : tt::sockets::transport_names::TCP;
}

volatile sig_atomic_t gStop = 0;

void signalHandler(int /*sig*/) { gStop = 1; }

std::vector<tt::gateway::GatewayPrefillMetricSnapshot> buildPrefillMetrics(
    const tt::gateway::PrefillRegistry& registry) {
  const auto now = std::chrono::steady_clock::now();
  std::vector<tt::gateway::GatewayPrefillMetricSnapshot> out;
  for (const auto& snapshot : registry.snapshot()) {
    double heartbeatAgeSeconds = 0.0;
    if (snapshot.last_heartbeat != std::chrono::steady_clock::time_point{}) {
      heartbeatAgeSeconds =
          std::chrono::duration<double>(now - snapshot.last_heartbeat).count();
    }
    out.push_back({snapshot.server_id, snapshot.healthy,
                   snapshot.accepting_tasks, snapshot.in_flight,
                   snapshot.cached_blocks, heartbeatAgeSeconds});
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  tt::utils::ZeroOverheadLogger::initialize();

  auto cfgOpt = parseArgs(argc, argv);
  if (!cfgOpt) return 1;
  const GatewayConfig& cfg = *cfgOpt;
  const bool useZmqPrefillRouter =
      socketTransportFromEnv() == tt::sockets::transport_names::ZMQ;
  const std::string_view transport = useZmqPrefillRouter ? "zmq" : "tcp";

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
  if (cfg.healthPort != 0 && cfg.healthPort == cfg.metricsPort) {
    std::cerr << "--health-port must be different from --metrics-port.\n";
    return 1;
  }

  TT_LOG_INFO("[Gateway] Starting — decode port={}, transport={}",
              cfg.decodePort, transport);

  auto& metrics = tt::gateway::GatewayMetrics::instance();
  tt::gateway::GatewayMetricsServer metricsServer(metrics);
  tt::gateway::GatewayHealthServer healthServer;

  tt::gateway::PrefillRegistry registry;
  tt::gateway::AffinityCache affinity;

  // Decode-facing: gateway listens, decode dials in (only 1 decode connection).
  tt::sockets::SocketManager decodeSm;
  if (!decodeSm.initializeAsServer(cfg.decodePort)) {
    TT_LOG_ERROR("[Gateway] Failed to bind decode port {}", cfg.decodePort);
    return 1;
  }

  auto healthProvider = [&registry, &decodeSm, transport]() {
    return buildGatewayHealthStatus(registry, transport,
                                    decodeSm.isConnected());
  };
  if (!metricsServer.start(cfg.metricsPort)) {
    TT_LOG_ERROR("[Gateway] Failed to start metrics endpoint on port {}",
                 cfg.metricsPort);
    return 1;
  }
  if (cfg.healthPort != 0) {
    healthServer.setHealthProvider(healthProvider);
    if (!healthServer.start(cfg.healthPort)) {
      TT_LOG_ERROR("[Gateway] Failed to start health endpoint on port {}",
                   cfg.healthPort);
      return 1;
    }
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
      sm->setReconnectBackoff(std::chrono::seconds(1), std::chrono::seconds(5));
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
      cfg.requestTimeout, cfg.timeoutWindow, cfg.timeoutCooldown,
      cfg.timeoutThreshold};
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

  decodeSm.registerHandler<tt::sockets::PrefillHealthRequestMessage>(
      tt::sockets::tags::PREFILL_HEALTH_REQUEST,
      [&decodeSm,
       &healthProvider](const tt::sockets::PrefillHealthRequestMessage&) {
        const auto health = healthProvider();
        tt::sockets::PrefillHealthStatusMessage response;
        response.ready = health.ready;
        (void)decodeSm.sendObject(tt::sockets::tags::PREFILL_HEALTH_STATUS,
                                  response);
      });

  decodeSm.setConnectionEstablishedCallback([&metrics]() {
    metrics.setDecodeConnected(true);
    TT_LOG_INFO("[Gateway] Decode connected");
  });
  decodeSm.setConnectionLostCallback([&metrics]() {
    metrics.setDecodeConnected(false);
    TT_LOG_WARN("[Gateway] Decode disconnected — waiting for reconnect");
  });

  for (auto& sm : prefillSms) sm->start();
  decodeSm.start();

  constexpr auto probeIntervalMs = std::chrono::milliseconds(1000);
  std::jthread proberThread(
      [&prefillSms, probeIntervalMs](std::stop_token stopToken) {
        while (!stopToken.stop_requested()) {
          for (auto& sm : prefillSms) {
            sm->sendObject(tt::sockets::tags::REGISTRATION_PROBE,
                           tt::sockets::RegistrationProbeMessage{});
          }
          std::this_thread::sleep_for(probeIntervalMs);
        }
      });

  const auto prefillStaleTimeout = cfg.prefillStaleTimeout;
  std::jthread watchdogThread;
  if (useZmqPrefillRouter) {
    watchdogThread =
        std::jthread([&registry, &zmqPrefillRouter,
                      prefillStaleTimeout](std::stop_token stopToken) {
          while (!stopToken.stop_requested()) {
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

  std::jthread requestTimeoutThread;
  if (cfg.requestTimeout.count() > 0) {
    requestTimeoutThread =
        std::jthread([&dispatcherPtr](std::stop_token stopToken) {
          while (!stopToken.stop_requested()) {
            dispatcherPtr->onRequestTimeouts();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
          }
        });
  }

  std::jthread metricsSnapshotThread(
      [&registry, &affinity, &metrics](std::stop_token stopToken) {
        while (!stopToken.stop_requested()) {
          metrics.setPrefillSnapshots(buildPrefillMetrics(registry));
          metrics.setRoutingTableSize(affinity.size());
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
      });

  TT_LOG_INFO("[Gateway] Running. Send SIGINT/SIGTERM to stop.");

  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  while (!gStop) {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  TT_LOG_INFO("[Gateway] Shutting down…");
  proberThread.request_stop();
  if (proberThread.joinable()) proberThread.join();
  requestTimeoutThread.request_stop();
  if (requestTimeoutThread.joinable()) requestTimeoutThread.join();
  metricsSnapshotThread.request_stop();
  if (metricsSnapshotThread.joinable()) metricsSnapshotThread.join();
  watchdogThread.request_stop();
  if (watchdogThread.joinable()) watchdogThread.join();
  decodeSm.stop();
  for (auto& sm : prefillSms) sm->stop();
  zmqPrefillRouter.stop();
  healthServer.stop();
  metricsServer.stop();
  TT_LOG_INFO("[Gateway] Stopped.");

  return 0;
}
