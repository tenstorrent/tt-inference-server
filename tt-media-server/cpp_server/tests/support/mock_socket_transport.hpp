// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Mock socket transport for disaggregation integration tests.
//
// Provides an in-process loopback transport that connects two SocketManagers
// directly, bypassing TCP/ZMQ. This allows testing the full prefill↔decode
// message flow without subprocess orchestration.
//
// Usage:
//   auto [decodeTransport, prefillTransport] = MockSocketTransport::createPair();
//   decodeSocketManager.setTransport(std::move(decodeTransport));
//   prefillSocketManager.setTransport(std::move(prefillTransport));
//
// Messages sent by one side are delivered to the other's receive queue.
// A message capture hook allows tests to inspect all traffic.

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "sockets/socket_messages.hpp"
#include "sockets/socket_serialization.hpp"

namespace tt::test {

// Forward declaration
class MockSocketTransport;

// Shared state between a pair of mock transports
struct MockTransportPair {
  std::mutex mutex;
  std::condition_variable cv;

  // Message queues: decode→prefill and prefill→decode
  std::deque<std::vector<uint8_t>> decodeToPrefill;
  std::deque<std::vector<uint8_t>> prefillToDecode;

  std::atomic<bool> connected{false};
  std::atomic<bool> stopped{false};

  // Capture hooks for test assertions
  std::function<void(const std::string& direction,
                     const std::vector<uint8_t>& data)>
      captureHook;
};

// Message capture for test assertions
struct CapturedSocketMessage {
  std::string direction;  // "decode→prefill" or "prefill→decode"
  std::string messageType;
  std::vector<uint8_t> rawData;

  // Parsed message fields (populated for known types)
  std::optional<sockets::PrefillRequestMessage> prefillRequest;
  std::optional<sockets::PrefillResultMessage> prefillResult;
};

class SocketMessageCapture {
 public:
  void capture(const std::string& direction, const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(mutex);
    CapturedSocketMessage msg;
    msg.direction = direction;
    msg.rawData = data;

    // Try to parse the message type and payload
    try {
      msg.messageType = sockets::wire::readMessageType(data);

      if (msg.messageType == "prefill_request") {
        msg.prefillRequest = sockets::wire::deserializePayload<
            sockets::PrefillRequestMessage>(data);
      } else if (msg.messageType == "prefill_result") {
        msg.prefillResult = sockets::wire::deserializePayload<
            sockets::PrefillResultMessage>(data);
      }
    } catch (...) {
      msg.messageType = "<parse_error>";
    }

    messages.push_back(std::move(msg));
    cv.notify_all();
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex);
    messages.clear();
  }

  std::vector<CapturedSocketMessage> getAll() const {
    std::lock_guard<std::mutex> lock(mutex);
    return messages;
  }

  std::vector<CapturedSocketMessage> getPrefillRequests() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<CapturedSocketMessage> result;
    for (const auto& m : messages) {
      if (m.prefillRequest.has_value()) {
        result.push_back(m);
      }
    }
    return result;
  }

  std::vector<CapturedSocketMessage> getPrefillResults() const {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<CapturedSocketMessage> result;
    for (const auto& m : messages) {
      if (m.prefillResult.has_value()) {
        result.push_back(m);
      }
    }
    return result;
  }

  bool waitForMessageCount(size_t count, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, timeout,
                        [this, count] { return messages.size() >= count; });
  }

 private:
  mutable std::mutex mutex;
  std::condition_variable cv;
  std::vector<CapturedSocketMessage> messages;
};

class MockSocketTransport : public sockets::ISocketTransport {
 public:
  enum class Role { DECODE, PREFILL };

  // Create a connected pair of transports
  static std::pair<std::unique_ptr<MockSocketTransport>,
                   std::unique_ptr<MockSocketTransport>>
  createPair(SocketMessageCapture* capture = nullptr) {
    auto shared = std::make_shared<MockTransportPair>();

    if (capture) {
      shared->captureHook = [capture](const std::string& dir,
                                      const std::vector<uint8_t>& data) {
        capture->capture(dir, data);
      };
    }

    auto decode =
        std::unique_ptr<MockSocketTransport>(new MockSocketTransport(
            shared, Role::DECODE));
    auto prefill =
        std::unique_ptr<MockSocketTransport>(new MockSocketTransport(
            shared, Role::PREFILL));

    return {std::move(decode), std::move(prefill)};
  }

  ~MockSocketTransport() override { stop(); }

  bool initializeAsServer(uint16_t /*port*/) override {
    // Mock transport doesn't need actual server setup
    return true;
  }

  bool initializeAsClient(const std::string& /*host*/,
                          uint16_t /*port*/) override {
    // Mock transport doesn't need actual client setup
    return true;
  }

  void start() override {
    shared->connected.store(true);
    shared->stopped.store(false);
    if (connectionEstablishedCallback) {
      connectionEstablishedCallback();
    }
  }

  void stop() override {
    shared->stopped.store(true);
    shared->connected.store(false);
    shared->cv.notify_all();
  }

  bool isConnected() const override { return shared->connected.load(); }

  std::string getStatus() const override {
    return isConnected() ? "connected" : "disconnected";
  }

  bool sendRawData(const std::vector<uint8_t>& data) override {
    if (!isConnected()) return false;

    std::lock_guard<std::mutex> lock(shared->mutex);

    // Capture the message if hook is set
    if (shared->captureHook) {
      std::string direction =
          (role == Role::DECODE) ? "decode→prefill" : "prefill→decode";
      shared->captureHook(direction, data);
    }

    // Route to the appropriate queue
    if (role == Role::DECODE) {
      shared->decodeToPrefill.push_back(data);
    } else {
      shared->prefillToDecode.push_back(data);
    }

    shared->cv.notify_all();
    return true;
  }

  std::vector<uint8_t> receiveRawData() override {
    std::unique_lock<std::mutex> lock(shared->mutex);

    // Select the queue we receive from (opposite of send direction)
    auto& queue = (role == Role::DECODE) ? shared->prefillToDecode
                                           : shared->decodeToPrefill;

    // Wait for data or stop signal
    shared->cv.wait(lock, [&] {
      return !queue.empty() || shared->stopped.load();
    });

    if (shared->stopped.load() && queue.empty()) {
      return {};
    }

    auto data = std::move(queue.front());
    queue.pop_front();
    return data;
  }

  void setConnectionLostCallback(std::function<void()> callback) override {
    connectionLostCallback = std::move(callback);
  }

  void setConnectionEstablishedCallback(
      std::function<void()> callback) override {
    connectionEstablishedCallback = std::move(callback);
  }

  // Test helper: simulate connection loss
  void simulateDisconnect() {
    shared->connected.store(false);
    if (connectionLostCallback) {
      connectionLostCallback();
    }
  }

  // Test helper: simulate reconnection
  void simulateReconnect() {
    shared->connected.store(true);
    if (connectionEstablishedCallback) {
      connectionEstablishedCallback();
    }
  }

 private:
  MockSocketTransport(std::shared_ptr<MockTransportPair> shared, Role role)
      : shared(std::move(shared)), role(role) {}

  std::shared_ptr<MockTransportPair> shared;
  Role role;
  std::function<void()> connectionLostCallback;
  std::function<void()> connectionEstablishedCallback;
};

}  // namespace tt::test
