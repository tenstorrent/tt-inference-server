// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "sockets/i_socket_transport.hpp"

// Forward-declare ZMQ types to avoid leaking zmq.hpp into every TU.
namespace zmq {
class context_t;
class socket_t;
}  // namespace zmq

namespace tt::sockets {

/**
 * @brief ZeroMQ-based socket transport for inter-server communication.
 *
 * Uses DEALER (client) / ROUTER (server) pattern for async bidirectional
 * messaging over tcp://. ZMQ handles framing, reconnection, and keepalive
 * internally — no manual length-prefix or retry logic needed.
 */
class ZmqSocketTransport : public ISocketTransport {
 public:
  ZmqSocketTransport();
  ZmqSocketTransport(const ZmqSocketTransport&) = delete;
  ZmqSocketTransport& operator=(const ZmqSocketTransport&) = delete;
  ~ZmqSocketTransport() override;

  bool initializeAsServer(uint16_t port) override;
  bool initializeAsClient(const std::string& host, uint16_t port) override;

  void start() override;
  void stop() override;

  bool isConnected() const override;
  std::string getStatus() const override;

  bool sendRawData(const std::vector<uint8_t>& data) override;
  std::vector<uint8_t> receiveRawData() override;

  void setConnectionLostCallback(std::function<void()> callback) override;

 private:
  enum class Mode { SERVER, CLIENT };

  void pollLoop();
  void monitorLoop();

  Mode mode_ = Mode::CLIENT;
  std::string endpoint_;

  std::unique_ptr<zmq::context_t> context_;
  std::unique_ptr<zmq::socket_t> socket_;

  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};

  std::thread pollThread_;
  std::thread monitorThread_;

  mutable std::mutex sendMutex_;
  mutable std::mutex peerIdMutex_;
  std::vector<uint8_t>
      peerId_;  // ROUTER stores the connected DEALER's identity.

  std::function<void()> connectionLostCallback_;
};

}  // namespace tt::sockets
