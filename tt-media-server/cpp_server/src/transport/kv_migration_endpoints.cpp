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

bool KvControlChannelConnector::openChannels() {
  bool allCreated = true;
  for (const auto& [host, endpoint] : endpoints_) {
    if (owned_.count(host) != 0) {
      continue;  // already created (idempotent re-open)
    }
    auto transport = factory_ ? factory_(endpoint) : nullptr;
    if (!transport) {
      TT_LOG_ERROR(
          "[KvControlChannelConnector] no transport for decode host '{}' "
          "({}:{})",
          host, endpoint.host, endpoint.port);
      allCreated = false;
      continue;
    }
    auto channel = std::make_unique<KvControlChannel>(
        std::move(transport), receive_timeout_, poll_interval_);
    channels_[host] = channel.get();
    owned_[host] = std::move(channel);
    TT_LOG_INFO("[KvControlChannelConnector] opened channel to '{}' ({}:{})",
                host, endpoint.host, endpoint.port);
  }
  return allCreated;
}

std::size_t KvControlChannelConnector::awaitConnected(
    std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  for (;;) {
    std::size_t connected = 0;
    for (const auto& [host, channel] : channels_) {
      if (channel->isConnected()) {
        ++connected;
      }
    }
    if (connected == channels_.size()) {
      return connected;  // all created channels are up (trivially true if none)
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return connected;  // timed out with some peers still connecting
    }
    std::this_thread::sleep_for(poll_interval_);
  }
}

std::unordered_map<std::string, KvControlChannel*>
KvControlChannelConnector::channels() const {
  return channels_;
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
      local_table_blob_(std::move(localTableBlob)),
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
  auto* orch = session->orchestrator.get();
  session->thread = std::thread([orch] { orch->run(); });

  std::size_t active = 0;
  {
    std::lock_guard<std::mutex> lock(sessionsMutex_);
    sessions_.push_back(std::move(session));
    active = sessions_.size();
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
