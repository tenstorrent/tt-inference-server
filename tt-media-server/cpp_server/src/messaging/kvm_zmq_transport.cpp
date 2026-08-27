// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "messaging/kvm_zmq_transport.hpp"

#include <cstring>
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

// ZMQ sockets are not thread-safe. `send()` (called from every prefill
// worker via `RemoteKVManagerZmqImpl::migrate`) must therefore be
// mutex-guarded around the PUB socket. `receive()` is only called from the
// drain thread inside `RemoteKVManagerZmqImpl`, so SUB is single-owner.
constexpr int LINGER_MS = 0;
constexpr int SNDHWM = 0;  // Unbounded — outbound bursts are 61 msgs.
constexpr int RCVHWM = 0;

}  // namespace

struct KvmZmqTransport::Impl {
  KvmZmqTransportConfig config;
  zmq::context_t context;
  zmq::socket_t pubSocket;
  zmq::socket_t subSocket;
  std::mutex sendMutex;

  Impl(KvmZmqTransportConfig cfg)
      : config(std::move(cfg)),
        context(/*io_threads=*/1),
        pubSocket(context, zmq::socket_type::pub),
        subSocket(context, zmq::socket_type::sub) {
    if (config.cmdEndpoint.empty()) {
      throw std::invalid_argument(
          "KvmZmqTransport: cmdEndpoint must be non-empty");
    }
    if (config.replyEndpoint.empty()) {
      throw std::invalid_argument(
          "KvmZmqTransport: replyEndpoint must be non-empty");
    }
    if (config.topic.empty()) {
      throw std::invalid_argument("KvmZmqTransport: topic must be non-empty");
    }

    pubSocket.set(zmq::sockopt::linger, LINGER_MS);
    pubSocket.set(zmq::sockopt::sndhwm, SNDHWM);

    subSocket.set(zmq::sockopt::linger, LINGER_MS);
    subSocket.set(zmq::sockopt::rcvhwm, RCVHWM);
    // Byte-prefix subscription — matches kv_manager's SUB filter model.
    subSocket.set(zmq::sockopt::subscribe, config.topic);

    // We bind (kv_manager connects). Order matters only for logs — SUB is
    // bound first so we're ready to receive the moment kv_manager's PUB
    // dials in.
    subSocket.bind(config.replyEndpoint);
    pubSocket.bind(config.cmdEndpoint);

    TT_LOG_INFO(
        "[KvmZmqTransport] bound cmd={} reply={} topic='{}' (kv_manager "
        "should connect its SUB->cmd and PUB->reply)",
        config.cmdEndpoint, config.replyEndpoint, config.topic);
  }

  ~Impl() {
    // Sockets close first (destructor order = reverse declaration), then
    // context. Both are RAII in cppzmq; explicit close is only needed if
    // we want to swallow errors, which we don't.
  }
};

KvmZmqTransport::KvmZmqTransport(KvmZmqTransportConfig config)
    : impl(std::make_unique<Impl>(std::move(config))) {}

KvmZmqTransport::~KvmZmqTransport() = default;

bool KvmZmqTransport::send(std::string_view payload,
                           std::string* errorMessage) {
  std::lock_guard<std::mutex> lock(impl->sendMutex);
  try {
    // Frame 1: topic prefix (SNDMORE). Frame 2: JSON payload. kv_manager's
    // SUB filters on frame 1 and echoes it back on the ack.
    zmq::message_t topicMsg(impl->config.topic.data(),
                            impl->config.topic.size());
    zmq::message_t payloadMsg(payload.data(), payload.size());

    auto topicRes = impl->pubSocket.send(topicMsg, zmq::send_flags::sndmore);
    if (!topicRes.has_value()) {
      if (errorMessage)
        *errorMessage = "KvmZmqTransport: topic send returned EAGAIN";
      return false;
    }
    auto payloadRes = impl->pubSocket.send(payloadMsg, zmq::send_flags::none);
    if (!payloadRes.has_value()) {
      if (errorMessage) {
        *errorMessage = "KvmZmqTransport: payload send returned EAGAIN";
      }
      return false;
    }
    return true;
  } catch (const zmq::error_t& e) {
    if (errorMessage) {
      *errorMessage = std::string("KvmZmqTransport: send failed: ") + e.what();
    }
    TT_LOG_ERROR("[KvmZmqTransport] send failed: {}", e.what());
    return false;
  }
}

std::optional<std::string> KvmZmqTransport::receive(int timeoutMs) {
  try {
    // Poll rather than setting ZMQ_RCVTIMEO so timeouts remain per-call
    // instead of sticky sockopt state (matters because the drain thread
    // wants different polling granularity than an ad-hoc caller might).
    zmq::pollitem_t items[] = {
        {static_cast<void*>(impl->subSocket), 0, ZMQ_POLLIN, 0}};
    const int rc = zmq::poll(items, 1, std::chrono::milliseconds(timeoutMs));
    if (rc <= 0 || (items[0].revents & ZMQ_POLLIN) == 0) {
      return std::nullopt;
    }

    // Frame 1: topic (we drop it — kv_manager echoes our topic back and it
    // adds no information the payload doesn't already carry via
    // command_id). Frame 2: JSON payload.
    zmq::message_t topicMsg;
    auto topicRes = impl->subSocket.recv(topicMsg, zmq::recv_flags::none);
    if (!topicRes.has_value()) {
      return std::nullopt;
    }

    std::string payload;
    if (topicMsg.more()) {
      zmq::message_t payloadMsg;
      auto payloadRes = impl->subSocket.recv(payloadMsg, zmq::recv_flags::none);
      if (!payloadRes.has_value()) {
        return std::nullopt;
      }
      payload.assign(static_cast<const char*>(payloadMsg.data()),
                     payloadMsg.size());
      // Defensive drain in case a peer ever sends >2 frames.
      while (payloadMsg.more()) {
        zmq::message_t extra;
        if (!impl->subSocket.recv(extra, zmq::recv_flags::none).has_value()) {
          break;
        }
        payloadMsg = std::move(extra);
      }
    } else {
      // Single-frame variant — treat the whole message as the payload with
      // the topic prefix stripped (mirrors kv_manager's tolerant SUB
      // path).
      std::string_view frame(static_cast<const char*>(topicMsg.data()),
                             topicMsg.size());
      if (frame.size() >= impl->config.topic.size() &&
          frame.compare(0, impl->config.topic.size(), impl->config.topic) ==
              0) {
        frame.remove_prefix(impl->config.topic.size());
      }
      payload.assign(frame);
    }

    return payload;
  } catch (const zmq::error_t& e) {
    // ETERM fires during context shutdown — normal, not an error.
    if (e.num() == ETERM) return std::nullopt;
    TT_LOG_ERROR("[KvmZmqTransport] receive failed: {}", e.what());
    return std::nullopt;
  }
}

}  // namespace tt::messaging
