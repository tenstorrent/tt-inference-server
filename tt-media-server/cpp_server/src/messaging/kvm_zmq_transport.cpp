// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "messaging/kvm_zmq_transport.hpp"

#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <zmq.hpp>

#include "utils/logger.hpp"

namespace tt::messaging {

namespace {

constexpr int LINGER_MS = 0;
constexpr int SNDHWM = 0;  // Unbounded — outbound bursts are 61 msgs.
constexpr int RCVHWM = 0;

}  // namespace

struct KvmZmqTransport::Impl {
  KvmZmqTransportConfig config;
  zmq::context_t context;
  zmq::socket_t dealerSocket;
  std::mutex queueMutex;
  std::deque<std::string> outbound;

  Impl(KvmZmqTransportConfig cfg)
      : config(std::move(cfg)),
        context(/*io_threads=*/1),
        dealerSocket(context, zmq::socket_type::dealer) {
    if (config.endpoint.empty()) {
      throw std::invalid_argument(
          "KvmZmqTransport: endpoint must be non-empty");
    }

    dealerSocket.set(zmq::sockopt::linger, LINGER_MS);
    dealerSocket.set(zmq::sockopt::sndhwm, SNDHWM);
    dealerSocket.set(zmq::sockopt::rcvhwm, RCVHWM);
    dealerSocket.connect(config.endpoint);

    TT_LOG_INFO("[KvmZmqTransport] connected DEALER to {}", config.endpoint);
  }
};

KvmZmqTransport::KvmZmqTransport(KvmZmqTransportConfig config)
    : impl(std::make_unique<Impl>(std::move(config))) {}

KvmZmqTransport::~KvmZmqTransport() = default;

bool KvmZmqTransport::send(std::string_view payload,
                           std::string* errorMessage) {
  try {
    std::lock_guard<std::mutex> lock(impl->queueMutex);
    impl->outbound.emplace_back(payload);
    return true;
  } catch (const std::exception& e) {
    if (errorMessage) {
      *errorMessage =
          std::string("KvmZmqTransport: enqueue failed: ") + e.what();
    }
    TT_LOG_ERROR("[KvmZmqTransport] enqueue failed: {}", e.what());
    return false;
  }
}

std::optional<std::string> KvmZmqTransport::receive(int timeoutMs) {
  try {
    std::deque<std::string> outbound;
    {
      std::lock_guard<std::mutex> lock(impl->queueMutex);
      outbound.swap(impl->outbound);
    }
    for (const std::string& payload : outbound) {
      zmq::message_t payloadMsg(payload.data(), payload.size());
      if (!impl->dealerSocket.send(payloadMsg, zmq::send_flags::none)
               .has_value()) {
        TT_LOG_ERROR("[KvmZmqTransport] payload send returned EAGAIN");
        return std::nullopt;
      }
    }

    zmq::pollitem_t items[] = {
        {static_cast<void*>(impl->dealerSocket), 0, ZMQ_POLLIN, 0}};
    const int rc = zmq::poll(items, 1, std::chrono::milliseconds(timeoutMs));
    if (rc <= 0 || (items[0].revents & ZMQ_POLLIN) == 0) {
      return std::nullopt;
    }

    zmq::message_t payloadMsg;
    auto payloadRes =
        impl->dealerSocket.recv(payloadMsg, zmq::recv_flags::none);
    if (!payloadRes.has_value()) {
      return std::nullopt;
    }
    if (payloadMsg.more()) {
      TT_LOG_ERROR("[KvmZmqTransport] received multipart ack");
      while (payloadMsg.more()) {
        zmq::message_t extra;
        if (!impl->dealerSocket.recv(extra, zmq::recv_flags::none)
                 .has_value()) {
          break;
        }
        payloadMsg = std::move(extra);
      }
      return std::nullopt;
    }
    return std::string(static_cast<const char*>(payloadMsg.data()),
                       payloadMsg.size());
  } catch (const zmq::error_t& e) {
    // ETERM fires during context shutdown — normal, not an error.
    if (e.num() == ETERM) return std::nullopt;
    TT_LOG_ERROR("[KvmZmqTransport] receive failed: {}", e.what());
    return std::nullopt;
  }
}

}  // namespace tt::messaging
