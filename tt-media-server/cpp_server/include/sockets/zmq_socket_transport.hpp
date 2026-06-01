// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "sockets/socket_transport_state.hpp"

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
class ZmqSocketTransport : public ISocketTransport,
                           protected SocketTransportState {
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

  bool sendRawData(std::span<const uint8_t> data) override;
  std::vector<uint8_t> receiveRawData() override;

  void setConnectionLostCallback(std::function<void()> callback) override;
  void setConnectionEstablishedCallback(
      std::function<void()> callback) override;
  void setReconnectBackoff(std::chrono::milliseconds initialDelay,
                           std::chrono::milliseconds maxDelay) override;

 private:
  struct SendRequest {
    std::vector<uint8_t> data;
    std::promise<bool> result;
  };

  struct SendQueue {
    std::mutex queueMutex;
    std::mutex wakeMutex;
    std::condition_variable wakeCv;
    std::atomic<bool> hasItems{false};
    std::deque<std::shared_ptr<SendRequest>> items;
  };

  bool startIoThread();
  void ioLoop(std::stop_token stopToken, std::promise<bool> initialized);
  bool initializeSocket();
  void setupMonitor();
  void monitorLoop(std::stop_token stopToken, std::promise<void> ready);
  bool processPendingSends();
  bool receiveAvailableMessages();
  void waitForIoWork();
  void failPendingSends();
  void enqueueReceivedMessage(std::vector<uint8_t> data);

  // Transport-specific send/receive halves. Only ioThread_ calls these methods;
  // no other thread touches socket_.
  bool sendAsRouter(const std::vector<uint8_t>& data);
  bool sendAsDealer(const std::vector<uint8_t>& data);
  std::vector<uint8_t> receiveAsRouter();
  std::vector<uint8_t> receiveAsDealer();

  std::string endpoint_;

  std::unique_ptr<zmq::context_t> context_;
  std::unique_ptr<zmq::socket_t> socket_;

  std::atomic<bool> ioActive_{false};
  std::atomic<bool> monitorActive_{false};

  std::jthread ioThread_;
  std::jthread monitorThread_;

  std::vector<uint8_t>
      peerId_;  // ROUTER stores the connected DEALER's identity.

  SendQueue sendQueue;

  std::mutex receiveMutex_;
  std::deque<std::vector<uint8_t>> receivedMessages_;
};

}  // namespace tt::sockets
