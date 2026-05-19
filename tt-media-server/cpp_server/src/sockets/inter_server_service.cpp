// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/inter_server_service.hpp"

#include <chrono>
#include <string>
#include <thread>

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
  const bool gatewayMode = tt::config::usePrefillGateway();

  bool success = false;

  // Gateway mode inverts roles: decode becomes CLIENT, prefill becomes SERVER,
  // and the gateway sits between them.
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    if (gatewayMode) {
      TT_LOG_INFO(
          "[InterServerService] Decode (gateway mode): connecting to {}:{}",
          host, port);
      success = socket_manager_.initializeAsClient(host, port);
    } else {
      TT_LOG_INFO(
          "[InterServerService] Decode (direct mode): listening on port {}",
          port);
      success = socket_manager_.initializeAsServer(port);
    }
  } else if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    if (gatewayMode) {
      TT_LOG_INFO(
          "[InterServerService] Prefill (gateway mode): listening on port {} "
          "for gateway",
          port);
      success = socket_manager_.initializeAsServer(port);
      gateway_mode_ = success;
    } else {
      TT_LOG_INFO(
          "[InterServerService] Prefill (direct mode): connecting to {}:{}",
          host, port);
      success = socket_manager_.initializeAsClient(host, port);
      direct_prefill_mode_ = success;
    }
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
  if (direct_prefill_mode_) {
    startDirectModeRegistrationThread();
  }
  TT_LOG_INFO("[InterServerService] Started socket communication");
}

void InterServerService::stop() {
  if (!enabled_) {
    return;
  }

  registration_running_ = false;
  if (registration_thread_.joinable()) {
    registration_thread_.join();
  }
  socket_manager_.stop();
  TT_LOG_INFO("[InterServerService] Stopped socket communication");
}

bool InterServerService::isEnabled() const { return enabled_; }

bool InterServerService::sendPrefillRequest(
    uint32_t taskId, size_t registrationHash,
    const std::vector<int64_t>& tokenIds, std::optional<int> maxTokens,
    std::optional<uint32_t> slotId,
    const tt::domain::llm::SamplingParams& sampling) {
  if (!enabled_) {
    return false;
  }

  PrefillRequestMessage message(taskId);
  message.registration_hash = registrationHash;
  message.token_ids = tokenIds;
  message.max_tokens = maxTokens;
  message.slot_id = slotId;
  message.temperature = sampling.temperature;
  message.top_p = sampling.top_p;
  message.top_k = sampling.top_k;
  message.fast_mode = sampling.fast_mode;

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

  // Decode-side no-op handler so the gateway's PrefillAssignment doesn't log
  // "no handler" warnings. Useful for KV-transfer routing in a follow-up.
  socket_manager_.registerHandler<PrefillAssignmentMessage>(
      tags::PREFILL_ASSIGNMENT, [](const PrefillAssignmentMessage& message) {
        TT_LOG_DEBUG(
            "[InterServerService] PrefillAssignment for task {} → prefill '{}'",
            message.task_id, message.server_id);
      });

  socket_manager_.registerHandler<RegistrationProbeMessage>(
      tags::REGISTRATION_PROBE, [this](const RegistrationProbeMessage&) {
        sendRegistrationIfGatewayModeIsEnabled();
      });

  socket_manager_.registerHandler<PrefillRegistrationMessage>(
      tags::PREFILL_REGISTRATION, [](const PrefillRegistrationMessage& msg) {
        TT_LOG_DEBUG(
            "[InterServerService] Prefill '{}' announced (direct mode)",
            msg.server_id);
      });

  // Handle incoming prefill results
  socket_manager_.registerHandler<PrefillResultMessage>(
      "prefill_result", [this](const PrefillResultMessage& message) {
        if (message.error) {
          TT_LOG_ERROR(
              "[InterServerService] Received prefill error for task: {}",
              message.task_id);
        } else {
          TT_LOG_INFO(
              "[InterServerService] Received prefill result: {} - text: '{}', "
              "remaining: {}, token_ids: {}",
              message.task_id, message.generated_text.substr(0, 50),
              message.remaining_tokens.has_value()
                  ? std::to_string(message.remaining_tokens.value())
                  : "none",
              message.token_ids.size());
        }
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

void InterServerService::sendRegistration() {
  PrefillRegistrationMessage msg;
  msg.server_id = tt::config::prefillServerId();
  msg.max_in_flight = tt::config::prefillMaxInFlight();

  bool ok = socket_manager_.sendObject(tags::PREFILL_REGISTRATION, msg);
  if (ok) {
    TT_LOG_DEBUG(
        "[InterServerService] Sent PrefillRegistration: id='{}' "
        "max_in_flight={}",
        msg.server_id, msg.max_in_flight);
  } else {
    TT_LOG_WARN("[InterServerService] Failed to send PrefillRegistration");
  }
}

void InterServerService::sendRegistrationIfGatewayModeIsEnabled() {
  if (!gateway_mode_) {
    return;
  }
  sendRegistration();
}

void InterServerService::startDirectModeRegistrationThread() {
  registration_running_ = true;
  registration_thread_ = std::thread([this] {
    while (registration_running_) {
      sendRegistration();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  });
}

}  // namespace tt::sockets
