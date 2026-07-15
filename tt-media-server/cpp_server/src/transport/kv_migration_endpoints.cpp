// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_migration_endpoints.hpp"

#include <utility>

#include "utils/logger.hpp"

namespace tt::transport {

// ---------------------------------------------------------------------------
// KvControlChannelConnector
// ---------------------------------------------------------------------------

KvControlChannelConnector::KvControlChannelConnector(
    std::unordered_map<std::string, Endpoint> endpoints,
    TransportFactory factory, std::chrono::milliseconds receiveTimeout,
    std::chrono::milliseconds pollInterval)
    : endpoints_(std::move(endpoints)),
      factory_(std::move(factory)),
      receive_timeout_(receiveTimeout),
      poll_interval_(pollInterval) {}

bool KvControlChannelConnector::openChannelLocked(const std::string& name,
                                                  const Endpoint& endpoint) {
  if (channels_.count(name) != 0) {
    return true;  // already created (idempotent)
  }
  auto transport = factory_ ? factory_(endpoint) : nullptr;
  if (!transport) {
    TT_LOG_ERROR(
        "[KvControlChannelConnector] no transport for decode host '{}' "
        "({}:{})",
        name, endpoint.host, endpoint.port);
    return false;
  }
  endpoints_[name] = endpoint;
  channels_[name] = std::make_shared<KvControlChannel>(
      std::move(transport), receive_timeout_, poll_interval_);
  TT_LOG_INFO("[KvControlChannelConnector] opened channel to '{}' ({}:{})",
              name, endpoint.host, endpoint.port);
  return true;
}

bool KvControlChannelConnector::openChannels() {
  std::lock_guard<std::mutex> lock(mutex_);
  bool allCreated = true;
  for (const auto& [host, endpoint] : endpoints_) {
    if (!openChannelLocked(host, endpoint)) {
      allCreated = false;
    }
  }
  return allCreated;
}

bool KvControlChannelConnector::openChannel(const std::string& name,
                                            const Endpoint& endpoint) {
  std::lock_guard<std::mutex> lock(mutex_);
  return openChannelLocked(name, endpoint);
}

bool KvControlChannelConnector::replaceChannelLocked(const std::string& name,
                                                     const Endpoint& endpoint) {
  // Drop the connector's strong ref. In-flight migrate() / mesh-watch holders
  // keep the old channel alive via their own shared_ptr copies.
  channels_.erase(name);
  endpoints_.erase(name);
  return openChannelLocked(name, endpoint);
}

bool KvControlChannelConnector::replaceChannel(const std::string& name,
                                               const Endpoint& endpoint) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = endpoints_.find(name);
  if (it != endpoints_.end() && it->second == endpoint &&
      channels_.count(name) != 0) {
    return true;  // already dialing / dialed this endpoint
  }
  TT_LOG_INFO("[KvControlChannelConnector] replacing channel to '{}' -> {}:{}",
              name, endpoint.host, endpoint.port);
  return replaceChannelLocked(name, endpoint);
}

std::size_t KvControlChannelConnector::awaitConnected(
    std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    std::size_t connected = 0;
    std::size_t total = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      total = channels_.size();
      for (const auto& [host, channel] : channels_) {
        if (channel->isConnected()) {
          ++connected;
        }
      }
    }
    if (connected == total) {
      return connected;  // all created channels are up (trivially true if none)
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return connected;  // timed out with some peers still connecting
    }
    std::this_thread::sleep_for(poll_interval_);
  }
}

std::unordered_map<std::string, std::shared_ptr<KvControlChannel>>
KvControlChannelConnector::channels() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return channels_;
}

std::optional<KvControlChannelConnector::Endpoint>
KvControlChannelConnector::endpoint(const std::string& name) const {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto it = endpoints_.find(name);
  if (it == endpoints_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::size_t KvControlChannelConnector::channelCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return channels_.size();
}

std::size_t KvControlChannelConnector::connectedCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::size_t connected = 0;
  for (const auto& [host, channel] : channels_) {
    if (channel->isConnected()) {
      ++connected;
    }
  }
  return connected;
}

// ---------------------------------------------------------------------------
// KvMigrationReceiverServer
// ---------------------------------------------------------------------------

KvMigrationReceiverServer::KvMigrationReceiverServer(
    uint16_t port, ServerTransportFactory factory, MooncakeKvReceiver& receiver,
    std::vector<uint8_t> localTableBlob,
    std::chrono::milliseconds receiveTimeout,
    std::chrono::milliseconds pollInterval)
    : port_(port),
      factory_(std::move(factory)),
      receiver_(receiver),
      local_table_blob_(std::make_shared<const std::vector<uint8_t>>(
          std::move(localTableBlob))),
      receive_timeout_(receiveTimeout),
      poll_interval_(pollInterval) {}

KvMigrationReceiverServer::~KvMigrationReceiverServer() { stop(); }

void KvMigrationReceiverServer::startSingleSession(
    std::shared_ptr<sockets::ISocketTransport> transport) {
  channel_ = std::make_unique<KvControlChannel>(
      std::move(transport), receive_timeout_, poll_interval_);
  orchestrator_ = std::make_unique<KvMigrationReceiver>(*channel_, receiver_,
                                                        local_table_blob_);
  thread_ = std::thread([this] { orchestrator_->run(); });
}

void KvMigrationReceiverServer::reapFinishedSessions() {
  // try_lock: stop() may hold sessionsMutex_ while joining us — never block
  // the session thread on that lock (deadlock). stop()/onAccept will reap.
  std::unique_lock<std::mutex> lock(sessionsMutex_, std::try_to_lock);
  if (!lock.owns_lock()) {
    return;
  }
  reapFinishedSessionsLocked();
}

void KvMigrationReceiverServer::reapFinishedSessionsLocked() {
  const auto selfId = std::this_thread::get_id();
  std::thread toDetach;
  for (auto it = sessions_.begin(); it != sessions_.end();) {
    if (!(*it)->finished.load(std::memory_order_acquire)) {
      ++it;
      continue;
    }
    // Session thread cannot join itself — detach so we can free the Session
    // (and its peer-table blob) without waiting for the next accept.
    if ((*it)->thread.joinable() && (*it)->thread.get_id() == selfId) {
      toDetach = std::move((*it)->thread);
      it = sessions_.erase(it);
      continue;
    }
    if ((*it)->thread.joinable()) {
      (*it)->thread.join();
    }
    it = sessions_.erase(it);
  }
  if (toDetach.joinable()) {
    toDetach.detach();
  }
}

std::size_t KvMigrationReceiverServer::activeSessionCount() const {
  std::lock_guard<std::mutex> lock(sessionsMutex_);
  std::size_t active = 0;
  for (const auto& session : sessions_) {
    if (!session->finished.load(std::memory_order_acquire)) {
      ++active;
    }
  }
  return active;
}

void KvMigrationReceiverServer::onAccept(
    std::shared_ptr<sockets::ISocketTransport> peer) {
  if (!peer) {
    TT_LOG_ERROR("[KvMigrationReceiverServer] null peer transport on port {}",
                 port_);
    return;
  }

  auto session = std::make_unique<Session>();
  session->transport = std::move(peer);
  session->channel = std::make_unique<KvControlChannel>(
      session->transport, receive_timeout_, poll_interval_);
  session->orchestrator = std::make_unique<KvMigrationReceiver>(
      *session->channel, receiver_, local_table_blob_);
  Session* raw = session.get();
  auto* orch = session->orchestrator.get();
  session->thread = std::thread([this, orch, raw] {
    orch->run();
    raw->finished.store(true, std::memory_order_release);
    // Free finished sessions (including this one) without waiting for another
    // prefill to connect — peer table blobs are large.
    reapFinishedSessions();
  });

  std::size_t active = 0;
  {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    reapFinishedSessionsLocked();
    sessions_.push_back(std::move(session));
    for (const auto& s : sessions_) {
      if (!s->finished.load(std::memory_order_acquire)) {
        ++active;
      }
    }
  }
  TT_LOG_INFO(
      "[KvMigrationReceiverServer] accepted prefill control session on :{} "
      "(active={})",
      port_, active);
}

bool KvMigrationReceiverServer::start() {
  if (running_) {
    return true;
  }
  listenTransport_ = factory_ ? factory_(port_) : nullptr;
  if (!listenTransport_) {
    TT_LOG_ERROR("[KvMigrationReceiverServer] no server transport on port {}",
                 port_);
    return false;
  }

  // Production TCP: multi-accept so every prefill gets a session.
  // Unit-test fakes: enableMultiAccept returns false — single connected peer.
  if (listenTransport_->enableMultiAccept(
          [this](std::shared_ptr<sockets::ISocketTransport> peer) {
            onAccept(std::move(peer));
          })) {
    listenTransport_->start();
  } else {
    startSingleSession(listenTransport_);
  }

  running_ = true;
  TT_LOG_INFO("[KvMigrationReceiverServer] listening on port {}", port_);
  return true;
}

void KvMigrationReceiverServer::stop() {
  if (!running_) {
    return;
  }
  running_ = false;

  if (listenTransport_) {
    listenTransport_->stop();
  }

  {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    for (auto& session : sessions_) {
      if (session->transport) {
        session->transport->stop();
      }
      if (session->thread.joinable()) {
        session->thread.join();
      }
    }
    sessions_.clear();
  }

  if (thread_.joinable()) {
    thread_.join();
  }
  orchestrator_.reset();
  channel_.reset();
  listenTransport_.reset();

  TT_LOG_INFO("[KvMigrationReceiverServer] stopped (port {})", port_);
}

}  // namespace tt::transport
