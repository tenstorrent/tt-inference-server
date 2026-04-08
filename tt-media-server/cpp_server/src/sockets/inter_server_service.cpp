// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "sockets/inter_server_service.hpp"

#include <string>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {

InterServerService::InterServerService() { setupMessageHandlers(); }

InterServerService::~InterServerService() { stop(); }

bool InterServerService::initializeFromConfig() {
  auto mode = tt::config::llmMode();

  if (mode == tt::config::LLMMode::REGULAR) {
    TT_LOG_INFO(
        "[InterServerService] Socket communication disabled (regular mode)");
    enabled_ = false;
    return false;
  }

  auto host = tt::config::socketHost();
  auto port = tt::config::socketPort();

  bool success = false;

  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    TT_LOG_INFO("[InterServerService] Initializing as server on port {}", port);
    success = socket_manager_.initializeAsServer(port);
  } else if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    TT_LOG_INFO("[InterServerService] Initializing as client to {}:{}", host,
                port);
    success = socket_manager_.initializeAsClient(host, port);
  }

  if (success) {
    enabled_ = true;
    TT_LOG_INFO(
        "[InterServerService] Socket communication initialized successfully");
  } else {
    enabled_ = false;
    TT_LOG_ERROR(
        "[InterServerService] Failed to initialize socket communication");
  }

  return success;
}

void InterServerService::start() {
  if (!enabled_) {
    return;
  }

  socket_manager_.start();
  TT_LOG_INFO("[InterServerService] Started socket communication");
}

void InterServerService::stop() {
  if (!enabled_) {
    return;
  }

  socket_manager_.stop();
  TT_LOG_INFO("[InterServerService] Stopped socket communication");
}

bool InterServerService::isEnabled() const { return enabled_; }

bool InterServerService::sendPrefillRequest(
    uint32_t taskId, const std::string& prompt,
    const std::vector<int64_t>& tokenIds, std::optional<int> maxTokens,
    std::optional<uint32_t> slotId) {
  if (!enabled_) {
    return false;
  }

  PrefillRequestMessage message(taskId);
  message.prompt = prompt;
  message.token_ids = tokenIds;
  message.max_tokens = maxTokens;
  message.slot_id = slotId;

  return socket_manager_.sendObject("prefill_request", message);
}

bool InterServerService::sendPrefillResult(
    const PrefillResultMessage& message) {
  if (!enabled_) {
    return false;
  }

  return socket_manager_.sendObject("prefill_result", message);
}

bool InterServerService::sendHealthCheck(const std::string& serverId,
                                         double cpuUsage, double memoryUsage,
                                         int activeTasks) {
  if (!enabled_) {
    return false;
  }

  HealthCheckMessage message;
  message.server_id = serverId;
  message.cpu_usage = cpuUsage;
  message.memory_usage = memoryUsage;
  message.active_tasks = activeTasks;

  return socket_manager_.sendObject("health_check", message);
}

void InterServerService::onPrefillRequested(PrefillRequestedCallback callback) {
  prefill_requested_callback_ = callback;
}

void InterServerService::onPrefillComplete(PrefillCompleteCallback callback) {
  prefill_complete_callback_ = callback;
}

void InterServerService::setHealthCheckCallback(HealthCallback callback) {
  health_check_callback_ = callback;
}

void InterServerService::setConnectionLostCallback(
    std::function<void()> callback) {
  socket_manager_.setConnectionLostCallback(std::move(callback));
}

bool InterServerService::isConnected() const {
  return enabled_ && socket_manager_.isConnected();
}

std::string InterServerService::getStatus() const {
  if (!enabled_) {
    return "disabled";
  }
  return socket_manager_.getStatus();
}

void InterServerService::setupMessageHandlers() {
  // Handle incoming prefill requests
  socket_manager_.registerHandler<PrefillRequestMessage>(
      "prefill_request", [this](const PrefillRequestMessage& message) {
        TT_LOG_INFO(
            "[InterServerService] Received prefill request: {} (tokens: {})",
            message.task_id, message.token_ids.size());
        if (prefill_requested_callback_) {
          prefill_requested_callback_(message);
        }
      });

  // Handle incoming prefill results
  socket_manager_.registerHandler<PrefillResultMessage>(
      "prefill_result", [this](const PrefillResultMessage& message) {
        TT_LOG_INFO(
            "[InterServerService] Received prefill result: {} - text: '{}', "
            "remaining: {}, token_ids: {}",
            message.task_id, message.generated_text.substr(0, 50),
            message.remaining_tokens.has_value()
                ? std::to_string(message.remaining_tokens.value())
                : "none",
            message.token_ids.size());
        if (prefill_complete_callback_) {
          prefill_complete_callback_(message);
        }
      });

  // Handle incoming health checks
  socket_manager_.registerHandler<HealthCheckMessage>(
      "health_check", [this](const HealthCheckMessage& message) {
        TT_LOG_DEBUG(
            "[InterServerService] Received health check from: {} (CPU: {}%, "
            "Memory: {}%, Tasks: {})",
            message.server_id, message.cpu_usage, message.memory_usage,
            message.active_tasks);
        if (health_check_callback_) {
          health_check_callback_(message.server_id, message.cpu_usage,
                                 message.memory_usage, message.active_tasks);
        }
      });
}

}  // namespace tt::sockets
