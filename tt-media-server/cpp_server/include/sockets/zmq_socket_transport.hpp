// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <functional>
#include <future>
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
  void setReconnectBackoff(uint32_t initialDelayMs,
                           uint32_t maxDelayMs) override;

 private:
  enum class Mode { SERVER, CLIENT };

  void setupMonitor();
  void monitorLoop(std::promise<void> ready);

  // Transport-specific send/receive halves. Each is called with socketMutex_
  // already held by the caller so they only touch socket_/peerId_ safely.
  bool sendAsRouter(const std::vector<uint8_t>& data);
  bool sendAsDealer(const std::vector<uint8_t>& data);
  std::vector<uint8_t> receiveAsRouter();
  std::vector<uint8_t> receiveAsDealer();

  Mode mode_ = Mode::CLIENT;
  std::string endpoint_;

  std::unique_ptr<zmq::context_t> context_;
  std::unique_ptr<zmq::socket_t> socket_;

  std::atomic<bool> running_{false};
  std::atomic<bool> connected_{false};
  std::atomic<bool> monitorActive_{false};

  std::thread monitorThread_;

  mutable std::mutex socketMutex_;
  mutable std::mutex peerIdMutex_;
  std::vector<uint8_t>
      peerId_;  // ROUTER stores the connected DEALER's identity.

  mutable std::mutex callbackMutex_;
  std::function<void()> connectionLostCallback_;

  // ZMQ_RECONNECT_IVL / ZMQ_RECONNECT_IVL_MAX, applied at initializeAsClient.
  uint32_t reconnectInitialDelayMs_{1000};
  uint32_t reconnectMaxDelayMs_{5000};
};

}  // namespace tt::sockets
