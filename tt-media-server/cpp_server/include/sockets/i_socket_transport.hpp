// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tt::sockets {

/**
 * @brief Wire-format names for SOCKET_TRANSPORT.
 */
namespace transport_names {
constexpr const char* TCP = "tcp";
constexpr const char* ZMQ = "zmq";
}  // namespace transport_names

/**
 * @brief Abstract interface for inter-server socket transports.
 *
 * Concrete implementations:
 *   - TcpSocketTransport  (raw POSIX TCP, the original implementation)
 *   - ZmqSocketTransport  (ZeroMQ DEALER/ROUTER over tcp://)
 *
 * Selectable at runtime via the SOCKET_TRANSPORT env var ("tcp" | "zmq").
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

  virtual bool sendRawData(const std::vector<uint8_t>& data) = 0;
  virtual std::vector<uint8_t> receiveRawData() = 0;

  virtual void setConnectionLostCallback(std::function<void()> callback) = 0;
  virtual void setConnectionEstablishedCallback(
      std::function<void()> callback) = 0;

  virtual void setReconnectBackoff(uint32_t /*initialDelayMs*/,
                                   uint32_t /*maxDelayMs*/) {}
};

/**
 * @brief Factory: creates the transport selected by config.
 */
std::unique_ptr<ISocketTransport> createSocketTransport();

}  // namespace tt::sockets
