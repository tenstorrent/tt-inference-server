// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "sockets/inter_server_service.hpp"

#include <chrono>
#include <string>

#include "config/settings.hpp"
#include "utils/logger.hpp"

namespace tt::sockets {
namespace {

constexpr auto REGISTRATION_INTERVAL = std::chrono::milliseconds(1000);
constexpr auto PREFILL_HEALTH_INTERVAL = std::chrono::milliseconds(1000);

}  // namespace

InterServerService::InterServerService() = default;

InterServerService::~InterServerService() { stop(); }

bool InterServerService::initializeFromConfig() {
  auto mode = tt::config::llmMode();
  llmMode = mode;

  if (mode == tt::config::LLMMode::REGULAR) {
    TT_LOG_INFO(
        "[InterServerService] Socket communication disabled (regular mode)");
    enabled = false;
    return false;
  }

  auto host = tt::config::socketHost();
  auto port = tt::config::socketPort();
  const bool useGatewayMode = tt::config::usePrefillGateway();

  bool success = false;

  // Gateway mode always makes decode dial the gateway.
  // ZMQ prefills dial the gateway's shared ROUTER socket.
  if (mode == tt::config::LLMMode::DECODE_ONLY) {
    if (useGatewayMode) {
      TT_LOG_INFO(
          "[InterServerService] Decode (gateway mode): connecting to {}:{}",
          host, port);
      success = socketManager.initializeAsClient(host, port);
    } else {
      TT_LOG_INFO(
          "[InterServerService] Decode (direct mode): listening on port {}",
          port);
      success = socketManager.initializeAsServer(port);
    }
    prefillHealthProbeMode = success;
  } else if (mode == tt::config::LLMMode::PREFILL_ONLY) {
    if (useGatewayMode) {
      TT_LOG_INFO(
          "[InterServerService] Prefill (gateway mode): connecting to {}:{}",
          host, port);
      success = socketManager.initializeAsClient(host, port);
      periodicRegistrationMode = success;
      gatewayMode = success;
    } else {
      TT_LOG_INFO(
          "[InterServerService] Prefill (direct mode): connecting to {}:{}",
          host, port);
      success = socketManager.initializeAsClient(host, port);
      // ZMQ needs one initial frame so decode can route back to prefill.
      periodicRegistrationMode = success;
    }
  }

  if (success) {
    enabled = true;
    setupMessageHandlers();
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

  if (periodicRegistrationMode) {
    // Register once on connect (and again on any reconnect). Gateway mode uses
    // this to keep the registry fresh; direct ZMQ mode uses it to expose the
    // DEALER identity to decode's ROUTER socket.
    socketManager.setConnectionEstablishedCallback(
        [this] { sendRegistration(); });
  }

  if (prefillHealthProbeMode) {
    markPrefillHealthUnavailable();
    socketManager.setConnectionEstablishedCallback([this] {
      markPrefillHealthUnavailable();
      sendPrefillHealthRequest();
    });
  }
  socketManager.start();
  if (periodicRegistrationMode && socketManager.isConnected()) {
    sendRegistration();
  }
  if (periodicRegistrationMode) {
    startRegistrationThread();
  }
  if (prefillHealthProbeMode) {
    startHealthProbeThread();
  }
  TT_LOG_INFO("[InterServerService] Started socket communication");
}

void InterServerService::stop() {
  if (!enabled) {
    return;
  }

  stopRegistrationThread();
  stopHealthProbeThread();
  socketManager.stop();
  TT_LOG_INFO("[InterServerService] Stopped socket communication");
}

bool InterServerService::isEnabled() const { return enabled; }

bool InterServerService::sendPrefillRequest(
    uint32_t taskId, const std::vector<uint64_t>& registrationHashes,
    const std::vector<uint32_t>& tokenIds, std::optional<int> maxTokens,
    std::optional<uint32_t> slotId,
    const tt::domain::llm::SamplingParams& sampling, int decodePositionId,
    int decodeSkipTokens) {
  if (!enabled) {
    return false;
  }

  PrefillRequestMessage message(taskId);
  message.registrationHashes = registrationHashes;
  message.tokenIds = tokenIds;
  message.maxTokens = maxTokens;
  message.slotId = slotId;
  message.temperature = sampling.temperature;
  message.topP = sampling.top_p;
  message.topK = sampling.top_k;
  message.fastMode = sampling.fast_mode;
  message.decodePositionId = decodePositionId;
  message.decodeSkipTokens = decodeSkipTokens;

  return socketManager.sendObject(tags::PREFILL_REQUEST, message);
}

bool InterServerService::sendPrefillResult(
    const PrefillResultMessage& message) {
  if (!enabled) {
    return false;
  }

  return socketManager.sendObject(tags::PREFILL_RESULT, message);
}

bool InterServerService::sendPrefillCancel(uint32_t taskId) {
  if (!enabled) {
    return false;
  }

  CancelPrefillMessage message;
  message.taskId = taskId;
  return socketManager.sendObject(tags::CANCEL_PREFILL, message);
}

bool InterServerService::sendPrefillCacheBlocksAdded(
    const std::vector<uint64_t>& blockHashes) {
  if (!enabled || !gatewayMode || blockHashes.empty()) {
    return false;
  }

  PrefillCacheBlocksAddedMessage message;
  message.serverId = tt::config::prefillServerId();
  message.blockHashes = blockHashes;
  return socketManager.sendObject(tags::PREFILL_CACHE_BLOCKS_ADDED, message);
}

void InterServerService::onPrefillRequested(PrefillRequestedCallback callback) {
  prefillRequestedCallback = callback;
}

void InterServerService::onPrefillCancelled(PrefillCancelCallback callback) {
  prefillCancelCallback = callback;
}

void InterServerService::onPrefillComplete(PrefillCompleteCallback callback) {
  prefillCompleteCallback = callback;
}

void InterServerService::setConnectionLostCallback(
    std::function<void()> callback) {
  socketManager.setConnectionLostCallback(
      [this, callback = std::move(callback)] {
        markPrefillHealthUnavailable();
        if (callback) {
          callback();
        }
      });
}

bool InterServerService::isConnected() const {
  return enabled && socketManager.isConnected() && isPrefillHealthReady();
}

std::string InterServerService::getStatus() const {
  if (!enabled) {
    return "disabled";
  }
  if (!prefillHealthProbeMode) {
    return socketManager.getStatus();
  }

  std::string status = socketManager.getStatus();
  status += isPrefillHealthReady() ? ", prefill_health=ready"
                                   : ", prefill_health=unavailable";
  return status;
}

void InterServerService::setupMessageHandlers() {
  if (llmMode == tt::config::LLMMode::PREFILL_ONLY) {
    socketManager.registerHandler<PrefillRequestMessage>(
        tags::PREFILL_REQUEST, [this](const PrefillRequestMessage& message) {
          TT_LOG_INFO(
              "[InterServerService] Received prefill request: {} (tokens: {})",
              message.taskId, message.tokenIds.size());
          if (prefillRequestedCallback) {
            prefillRequestedCallback(message);
          }
        });

    socketManager.registerHandler<CancelPrefillMessage>(
        tags::CANCEL_PREFILL, [this](const CancelPrefillMessage& message) {
          TT_LOG_INFO("[InterServerService] Received prefill cancel: {}",
                      message.taskId);
          if (prefillCancelCallback) {
            prefillCancelCallback(message);
          }
        });

    socketManager.registerHandler<RegistrationProbeMessage>(
        tags::REGISTRATION_PROBE, [this](const RegistrationProbeMessage&) {
          sendRegistrationIfGatewayModeIsEnabled();
        });

    socketManager.registerHandler<PrefillHealthRequestMessage>(
        tags::PREFILL_HEALTH_REQUEST,
        [this](const PrefillHealthRequestMessage&) {
          sendPrefillHealthStatus();
        });
  }

  if (llmMode == tt::config::LLMMode::DECODE_ONLY) {
    socketManager.registerHandler<PrefillHealthStatusMessage>(
        tags::PREFILL_HEALTH_STATUS,
        [this](const PrefillHealthStatusMessage& message) {
          recordPrefillHealthStatus(message);
        });

    socketManager.registerHandler<PrefillRegistrationMessage>(
        tags::PREFILL_REGISTRATION, [](const PrefillRegistrationMessage&) {});

    socketManager.registerHandler<PrefillResultMessage>(
        tags::PREFILL_RESULT, [this](const PrefillResultMessage& message) {
          if (message.error) {
            TT_LOG_ERROR(
                "[InterServerService] Received prefill error for task: {}",
                message.taskId);
          } else {
            TT_LOG_INFO(
                "[InterServerService] Received prefill result: {} - text: "
                "'{}', "
                "remaining: {}, token_ids: {}",
                message.taskId, message.generatedText.substr(0, 50),
                message.remainingTokens.has_value()
                    ? std::to_string(message.remainingTokens.value())
                    : "none",
                message.tokenIds.size());
          }
          if (prefillCompleteCallback) {
            prefillCompleteCallback(message);
          }
        });
  }
}

void InterServerService::sendRegistration() {
  PrefillRegistrationMessage msg;
  msg.serverId = tt::config::prefillServerId();
  msg.maxInFlight = tt::config::prefillMaxInFlight();

  bool ok = socketManager.sendObject(tags::PREFILL_REGISTRATION, msg);
  if (!ok) {
    TT_LOG_WARN("[InterServerService] Failed to send PrefillRegistration");
  }
}

void InterServerService::sendRegistrationIfGatewayModeIsEnabled() {
  if (!gatewayMode) {
    return;
  }
  sendRegistration();
}

void InterServerService::sendPrefillHealthRequest() {
  if (!prefillHealthProbeMode || !socketManager.isConnected()) {
    markPrefillHealthUnavailable();
    return;
  }

  PrefillHealthRequestMessage msg;
  const bool ok = socketManager.sendObject(tags::PREFILL_HEALTH_REQUEST, msg);
  if (!ok) {
    markPrefillHealthUnavailable();
  }
}

void InterServerService::sendPrefillHealthStatus() {
  if (llmMode != tt::config::LLMMode::PREFILL_ONLY ||
      !socketManager.isConnected()) {
    return;
  }

  PrefillHealthStatusMessage status;
  status.ready = true;
  (void)socketManager.sendObject(tags::PREFILL_HEALTH_STATUS, status);
}

void InterServerService::recordPrefillHealthStatus(
    const PrefillHealthStatusMessage& message) {
  if (!prefillHealthProbeMode) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(prefillHealthMutex);
    prefillHealthReady = message.ready;
  }
  prefillHealthCv.notify_all();

  if (!message.ready) {
    TT_LOG_WARN("[InterServerService] Prefill health unavailable");
  }
}

void InterServerService::markPrefillHealthUnavailable() {
  if (!prefillHealthProbeMode) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(prefillHealthMutex);
    prefillHealthReady = false;
  }
  prefillHealthCv.notify_all();
}

bool InterServerService::isPrefillHealthReady() const {
  if (!prefillHealthProbeMode) {
    return true;
  }

  std::lock_guard<std::mutex> lock(prefillHealthMutex);
  return prefillHealthReady;
}

void InterServerService::startRegistrationThread() {
  stopRegistrationThread();

  registrationThread = std::jthread([this](std::stop_token stopToken) {
    while (!stopToken.stop_requested()) {
      if (socketManager.isConnected()) {
        sendRegistration();
      }

      std::unique_lock<std::mutex> lock(registrationMutex);
      registrationCv.wait_for(lock, REGISTRATION_INTERVAL, [&stopToken] {
        return stopToken.stop_requested();
      });
    }
  });
}

void InterServerService::stopRegistrationThread() {
  registrationThread.request_stop();
  registrationCv.notify_all();
  if (registrationThread.joinable()) {
    registrationThread.join();
  }
}

void InterServerService::startHealthProbeThread() {
  stopHealthProbeThread();

  prefillHealthThread = std::jthread([this](std::stop_token stopToken) {
    while (!stopToken.stop_requested()) {
      sendPrefillHealthRequest();

      std::unique_lock<std::mutex> lock(prefillHealthMutex);
      prefillHealthCv.wait_for(lock, PREFILL_HEALTH_INTERVAL, [&stopToken] {
        return stopToken.stop_requested();
      });
    }
  });
}

void InterServerService::stopHealthProbeThread() {
  prefillHealthThread.request_stop();
  prefillHealthCv.notify_all();
  if (prefillHealthThread.joinable()) {
    prefillHealthThread.join();
  }
}

}  // namespace tt::sockets
