// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace tt::sockets {

/**
 * @brief Outcome of a non-blocking message receive.
 *
 * Unlike receiveRawData(), this keeps "no message ready yet" distinct from
 * "connection closed" so callers (e.g. KvControlChannel) can wait/retry on the
 * former instead of aborting.
 */
enum class ReceiveStatus : uint8_t {
  DATA,  ///< A complete message was received (payload in ReceiveResult::data).
  NO_DATA,  ///< Connection is live but no full message is buffered yet.
  CLOSED,   ///< Connection is closed or broken.
};

struct ReceiveResult {
  ReceiveStatus status = ReceiveStatus::CLOSED;
  std::vector<uint8_t> data;  ///< Non-empty only when status == DATA.
};

/**
 * @brief Abstract interface for inter-server socket transports.
 *
 * Implemented by ZmqSocketTransport (ZeroMQ DEALER/ROUTER over tcp://).
 */
class ISocketTransport {
 public:
  virtual ~ISocketTransport() = default;

  virtual bool initializeAsServer(uint16_t port) = 0;
  virtual bool initializeAsClient(const std::string& host, uint16_t port) = 0;

  virtual void start() = 0;
  virtual void stop() = 0;

  virtual bool isConnected() const = 0;
  virtual std::string getStatus() const = 0;

  virtual bool sendRawData(std::span<const uint8_t> data) = 0;

  /**
   * @brief Ownership-transfer send: hands the payload buffer to the transport.
   *
   * Lets transports avoid copying large payloads (e.g. pass the buffer straight
   * to a zero-copy zmq::message_t) on the hot decode->prefill path. The default
   * copies via the span overload, so transports that don't care need no change.
   */
  virtual bool sendRawData(std::vector<uint8_t>&& data) {
    return sendRawData(std::span<const uint8_t>(data.data(), data.size()));
  }

  virtual std::vector<uint8_t> receiveRawData() = 0;

  /**
   * @brief Non-blocking receive that distinguishes NO_DATA from CLOSED.
   *
   * receiveRawData() collapses "no message ready yet" and "connection closed"
   * into the same empty buffer, forcing callers to guess (e.g. by polling
   * isConnected(), which races a normal reply). This reports the three cases
   * explicitly so a control channel can wait/retry on NO_DATA and only abort on
   * CLOSED.
   *
   * The default delegates to receiveRawData() and can therefore only infer
   * NO_DATA vs CLOSED from isConnected(). Transports that know the difference
   * natively can override this to report the real status directly.
   */
  virtual ReceiveResult tryReceiveMessage() {
    std::vector<uint8_t> bytes = receiveRawData();
    if (!bytes.empty()) {
      return {ReceiveStatus::DATA, std::move(bytes)};
    }
    return {isConnected() ? ReceiveStatus::NO_DATA : ReceiveStatus::CLOSED, {}};
  }

  virtual void setConnectionLostCallback(std::function<void()> callback) = 0;
  virtual void setConnectionEstablishedCallback(
      std::function<void()> callback) = 0;

  virtual void setReconnectBackoff(std::chrono::milliseconds /*initialDelay*/,
                                   std::chrono::milliseconds /*maxDelay*/) {}
};

}  // namespace tt::sockets
