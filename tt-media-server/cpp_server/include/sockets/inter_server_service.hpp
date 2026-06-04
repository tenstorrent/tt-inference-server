// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "domain/llm/sampling_params.hpp"
#include "sockets/socket_manager.hpp"
#include "sockets/socket_messages.hpp"

namespace tt::sockets {

/**
 * @brief Service for managing inter-server communication in prefill/decode
 * split mode
 *
 * Handles prefill requests/results and health checks between prefill and decode
 * server instances. The decode server sends prefill requests to the prefill
 * server, which processes them and returns prefill results.
 */
class InterServerService {
 public:
  /**
   * @brief Callback type for when prefill completes
   */
  using PrefillCompleteCallback =
      std::function<void(const PrefillResultMessage& result)>;

  /**
   * @brief Callback type for when prefill is requested
   */
  using PrefillRequestedCallback =
      std::function<void(const PrefillRequestMessage& message)>;

  using PrefillCancelCallback =
      std::function<void(const CancelPrefillMessage& message)>;

  /**
   * @brief Health info callback type
   */
  using HealthCallback = std::function<void(
      const std::string& serverId, double cpu, double memory, int activeTasks)>;

  InterServerService();
  ~InterServerService();

  /**
   * @brief Initialize based on configuration settings
   * @return true if socket communication is enabled and initialized
   */
  bool initializeFromConfig();

  /**
   * @brief Start the inter-server communication (if enabled)
   */
  void start();

  /**
   * @brief Stop the inter-server communication
   */
  void stop();

  /**
   * @brief Check if socket communication is enabled
   */
  bool isEnabled() const;

  /**
   * @brief Send prefill request to the prefill server
   * @param task_id Unique task identifier
   * @param registrationHashes Prefix-cache block hashes for the conversation
   * @param token_ids Pre-tokenized prompt token IDs
   * @param max_tokens Maximum tokens to generate (nullopt = run until EOS)
   * @param slot_id KV cache slot allocated by decode server's memory manager
   * @param sampling Sampling parameters; only the subset carried on the wire
   *                 (temperature, top_p, top_k, fast_mode) is used. Pass the
   *                 result of mapSamplingParams() so global overrides like
   *                 USE_FAST_MODE are honoured. Defaulted SamplingParams{}
   *                 means "use prefill-side defaults".
   * @return true if sent successfully
   */
  bool sendPrefillRequest(uint32_t taskId,
                          const std::vector<uint64_t>& registrationHashes,
                          const std::vector<int64_t>& tokenIds,
                          std::optional<int> maxTokens = std::nullopt,
                          std::optional<uint32_t> slotId = std::nullopt,
                          const tt::domain::llm::SamplingParams& sampling = {},
                          int numberOfDecodeSkipTokens = 0,
                          const std::string& traceId = "",
                          const std::string& sessionId = "");

  /**
   * @brief Send prefill result back to the decode server
   * @param message Pre-built PrefillResultMessage
   * @return true if sent successfully
   */
  bool sendPrefillResult(const PrefillResultMessage& message);

  /**
   * @brief Best-effort cancellation for an in-flight prefill task.
   * @return true if sent successfully
   */
  bool sendPrefillCancel(uint32_t taskId);

  /**
   * @brief Send health check information
   * @param server_id This server's identifier
   * @param cpu_usage CPU usage percentage
   * @param memory_usage Memory usage percentage
   * @param active_tasks Number of active tasks
   * @return true if sent successfully
   */
  bool sendHealthCheck(const std::string& serverId, double cpuUsage,
                       double memoryUsage, int activeTasks);

  /**
   * @brief Set callback for when prefill server receives a request
   * @param callback Function to call when prefill request is received
   */
  void onPrefillRequested(PrefillRequestedCallback callback);

  /**
   * @brief Set callback for when prefill server receives a cancellation.
   */
  void onPrefillCancelled(PrefillCancelCallback callback);

  /**
   * @brief Set callback for when decode server receives prefill completion
   * @param callback Function to call when prefill is complete
   */
  void onPrefillComplete(PrefillCompleteCallback callback);

  /**
   * @brief Set callback for received health checks
   * @param callback Function to call when health info is received
   */
  void setHealthCheckCallback(HealthCallback callback);

  /**
   * @brief Set callback for connection lost events
   * @param callback Function to call when connection is lost
   */
  void setConnectionLostCallback(std::function<void()> callback);

  /**
   * @brief Check if connected to peer server
   */
  bool isConnected() const;

  /**
   * @brief Get connection status
   */
  std::string getStatus() const;

 private:
  void setupMessageHandlers();

  // Send PrefillRegistrationMessage to the peer (gateway or decode).
  void sendRegistration();

  // Prefill-side, gateway-mode only: send PrefillRegistrationMessage in
  // response to a RegistrationProbeMessage from the gateway. No-op otherwise.
  void sendRegistrationIfGatewayModeIsEnabled();
  void startRegistrationThread();
  void stopRegistrationThread();

  SocketManager socket_manager_;
  PrefillRequestedCallback prefill_requested_callback_;
  PrefillCancelCallback prefill_cancel_callback_;
  PrefillCompleteCallback prefill_complete_callback_;
  HealthCallback health_check_callback_;
  bool enabled_ = false;
  bool gateway_mode_ = false;
  bool periodic_registration_mode_ = false;
  std::mutex registration_mutex_;
  std::condition_variable registration_cv_;
  std::jthread registration_thread_;
};

}  // namespace tt::sockets
