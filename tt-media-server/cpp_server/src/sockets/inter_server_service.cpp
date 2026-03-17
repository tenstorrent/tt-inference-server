// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "sockets/inter_server_service.hpp"

#include <string>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {

InterServerService::InterServerService()
    : socketManager(SocketManager::getInstance()) {
  setupMessageHandlers();
}

InterServerService::~InterServerService() { stop(); }

bool InterServerService::initializeFromConfig() {
  auto mode = tt::config::llmMode();

  if (mode == tt::config::LLMMode::REGULAR) {
    TT_LOG_INFO(
        "[InterServerService] Socket communication disabled (regular mode)");
    enabled = false;
    return false;
  }

  auto host = tt::config::socketHost();
  auto port = tt::config::socketPort();

  bool success = false;

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    TT_LOG_INFO("[InterServerService] Initializing as server on port {}", port);
    success = socketManager.initializeAsServer(port);
  } else if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    TT_LOG_INFO("[InterServerService] Initializing as client to {}:{}", host,
                port);
    success = socketManager.initializeAsClient(host, port);
  }

  if (success) {
    enabled = true;
    TT_LOG_INFO(
        "[InterServerService] Socket communication initialized successfully");
  } else {
    enabled = false;
    TT_LOG_ERROR(
        "[InterServerService] Failed to initialize socket communication");
  }

  return success;
}

void InterServerService::start() {
  if (!enabled) {
    return;
  }

  socketManager.start();
  TT_LOG_INFO("[InterServerService] Started socket communication");
}

void InterServerService::stop() {
  if (!enabled) {
    return;
  }

  socketManager.stop();
  TT_LOG_INFO("[InterServerService] Stopped socket communication");
}

bool InterServerService::isEnabled() const { return enabled; }

bool InterServerService::sendPrefillRequest(
    const tt::domain::TaskID& taskId, const std::string& prompt,
    const std::vector<int64_t>& tokenIds, std::optional<int> maxTokens) {
  if (!enabled) {
    return false;
  }

  PrefillRequestMessage message(taskId);
  message.prompt = prompt;
  message.token_ids = tokenIds;
  message.max_tokens = maxTokens;

  return socketManager.sendObject("prefill_request", message);
}

bool InterServerService::sendPrefillResult(
    const PrefillResultMessage& message) {
  if (!enabled) {
    return false;
  }

  return socketManager.sendObject("prefill_result", message);
}

bool InterServerService::sendHealthCheck(const std::string& serverId,
                                         double cpuUsage, double memoryUsage,
                                         int activeTasks) {
  if (!enabled) {
    return false;
  }

  HealthCheckMessage message;
  message.server_id = serverId;
  message.cpu_usage = cpuUsage;
  message.memory_usage = memoryUsage;
  message.active_tasks = activeTasks;

  return socketManager.sendObject("health_check", message);
}

void InterServerService::onPrefillRequested(PrefillRequestedCallback callback) {
  prefillRequestedCallback = callback;
}

void InterServerService::onPrefillComplete(PrefillCompleteCallback callback) {
  prefillCompleteCallback = callback;
}

void InterServerService::setHealthCheckCallback(HealthCallback callback) {
  healthCheckCallback = callback;
}

void InterServerService::setConnectionLostCallback(
    std::function<void()> callback) {
  socketManager.setConnectionLostCallback(std::move(callback));
}

bool InterServerService::isConnected() const {
  return enabled && socketManager.isConnected();
}

std::string InterServerService::getStatus() const {
  if (!enabled) {
    return "disabled";
  }
  return socketManager.getStatus();
}

void InterServerService::setupMessageHandlers() {
  // Handle incoming prefill requests
  socketManager.registerHandler<PrefillRequestMessage>(
      "prefill_request", [this](const PrefillRequestMessage& message) {
        TT_LOG_INFO(
            "[InterServerService] Received prefill request: {} (tokens: {})",
            message.task_id.id, message.token_ids.size());
        if (prefillRequestedCallback) {
          prefillRequestedCallback(message);
        }
      });

  // Handle incoming prefill results
  socketManager.registerHandler<PrefillResultMessage>(
      "prefill_result", [this](const PrefillResultMessage& message) {
        TT_LOG_INFO(
            "[InterServerService] Received prefill result: {} - text: '{}', "
            "remaining: {}, token_ids: {}",
            message.task_id.id, message.generated_text.substr(0, 50),
            message.remaining_tokens.has_value()
                ? std::to_string(message.remaining_tokens.value())
                : "none",
            message.token_ids.size());
        if (prefillCompleteCallback) {
          prefillCompleteCallback(message);
        }
      });

  // Handle incoming health checks
  socketManager.registerHandler<HealthCheckMessage>(
      "health_check", [this](const HealthCheckMessage& message) {
        TT_LOG_DEBUG(
            "[InterServerService] Received health check from: {} (CPU: {}%, "
            "Memory: {}%, Tasks: {})",
            message.server_id, message.cpu_usage, message.memory_usage,
            message.active_tasks);
        if (healthCheckCallback) {
          healthCheckCallback(message.server_id, message.cpu_usage,
                                 message.memory_usage, message.active_tasks);
        }
      });
}

}  // namespace tt::sockets
