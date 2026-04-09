// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

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
   * @param prompt Task prompt (text)
   * @param token_ids Pre-tokenized prompt token IDs
   * @param max_tokens Maximum tokens to generate (nullopt = run until EOS)
   * @param slot_id KV cache slot allocated by decode server's memory manager
   * @return true if sent successfully
   */
  bool sendPrefillRequest(uint32_t taskId, const std::string& prompt,
                          const std::vector<int64_t>& tokenIds,
                          std::optional<int> maxTokens = std::nullopt,
                          std::optional<uint32_t> slotId = std::nullopt);

  /**
   * @brief Send prefill result back to the decode server
   * @param message Pre-built PrefillResultMessage
   * @return true if sent successfully
   */
  bool sendPrefillResult(const PrefillResultMessage& message);

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

  SocketManager socket_manager_;
  PrefillRequestedCallback prefill_requested_callback_;
  PrefillCompleteCallback prefill_complete_callback_;
  HealthCallback health_check_callback_;
  bool enabled_ = false;
};

}  // namespace tt::sockets
