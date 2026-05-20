// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "sockets/socket_serialization.hpp"

namespace tt::gateway {

class ZmqPrefillRouter {
 public:
  using PeerIdentity = std::vector<uint8_t>;

  ZmqPrefillRouter();
  ZmqPrefillRouter(const ZmqPrefillRouter&) = delete;
  ZmqPrefillRouter& operator=(const ZmqPrefillRouter&) = delete;
  ~ZmqPrefillRouter();

  bool start(const std::string& bindHost, uint16_t port);
  void stop();

  template <typename T>
  bool sendObject(const std::string& serverId, const std::string& messageType,
                  const T& obj);

  template <typename T>
  void registerHandler(
      const std::string& messageType,
      std::function<void(const PeerIdentity&, const T&)> handler);

  void rememberRegistration(const std::string& serverId,
                            const PeerIdentity& peerId);
  std::optional<std::string> serverIdForPeer(const PeerIdentity& peerId) const;
  std::vector<std::string> takeStaleServers(std::chrono::milliseconds timeout);

 private:
  struct SendRequest {
    std::string peerKey;
    std::vector<uint8_t> data;
    std::promise<bool> result;
  };

  using RawHandler =
      std::function<void(const PeerIdentity&, const std::vector<uint8_t>&)>;

  static std::string peerKey(const PeerIdentity& peerId);

  bool startIoThread();
  void ioLoop(std::promise<bool> initialized);
  bool initializeSocket();
  bool processPendingSends();
  bool receiveAvailableMessages();
  void waitForIoWork();
  void failPendingSends();
  void handleIncomingMessage(const PeerIdentity& peerId,
                             const std::vector<uint8_t>& data);
  std::optional<PeerIdentity> peerIdForServer(
      const std::string& serverId) const;

  std::string endpoint_;

  class Impl;
  std::unique_ptr<Impl> impl_;

  std::atomic<bool> running_{false};
  std::thread io_thread_;

  mutable std::mutex peer_mutex_;
  std::unordered_map<std::string, PeerIdentity> server_to_peer_;
  std::unordered_map<std::string, std::string> peer_to_server_;
  std::unordered_map<std::string, std::chrono::steady_clock::time_point>
      last_seen_by_server_;

  std::mutex send_mutex_;
  std::condition_variable send_cv_;
  std::deque<std::shared_ptr<SendRequest>> pending_sends_;

  mutable std::mutex handlers_mutex_;
  std::unordered_map<std::string, RawHandler> handlers_;
};

template <typename T>
bool ZmqPrefillRouter::sendObject(const std::string& serverId,
                                  const std::string& messageType,
                                  const T& obj) {
  auto peerId = peerIdForServer(serverId);
  if (!peerId.has_value()) {
    return false;
  }

  try {
    auto request = std::make_shared<SendRequest>();
    request->peerKey = peerKey(*peerId);
    request->data = tt::sockets::wire::serializeMessage(messageType, obj);
    auto result = request->result.get_future();

    {
      std::lock_guard<std::mutex> lock(send_mutex_);
      if (!running_) {
        return false;
      }
      pending_sends_.push_back(std::move(request));
    }
    send_cv_.notify_one();
    return result.get();
  } catch (const std::exception&) {
    return false;
  }
}

template <typename T>
void ZmqPrefillRouter::registerHandler(
    const std::string& messageType,
    std::function<void(const PeerIdentity&, const T&)> handler) {
  std::lock_guard<std::mutex> lock(handlers_mutex_);
  handlers_[messageType] = [handler](const PeerIdentity& peerId,
                                     const std::vector<uint8_t>& data) {
    T payload = tt::sockets::wire::deserializePayload<T>(data);
    handler(peerId, payload);
  };
}

}  // namespace tt::gateway
