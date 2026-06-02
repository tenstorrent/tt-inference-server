// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// End-to-end integration test for PrefillGateway over real loopback sockets.
// Validates registration handshake, routing (round-robin + sticky-by-hash),
// result/assignment delivery, and prefill-down failover.

#include <arpa/inet.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <zmq.hpp>

#include "gateway/affinity_cache.hpp"
#include "gateway/dispatcher.hpp"
#include "gateway/gateway_metrics.hpp"
#include "gateway/gateway_metrics_server.hpp"
#include "gateway/prefill_registry.hpp"
#include "gateway/zmq_prefill_router.hpp"
#include "sockets/socket_manager.hpp"
#include "sockets/socket_messages.hpp"
#include "sockets/socket_serialization.hpp"
#include "sockets/zmq_socket_options.hpp"
#include "sockets/zmq_socket_transport.hpp"

namespace tt::gateway {
namespace {

using namespace std::chrono_literals;

template <typename Pred>
bool waitFor(Pred pred, std::chrono::milliseconds timeout = 2s) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) return true;
    std::this_thread::sleep_for(10ms);
  }
  return pred();
}

uint16_t ephemeralPort() {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(s, 0);
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = 0;
  EXPECT_EQ(bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  socklen_t len = sizeof(addr);
  EXPECT_EQ(getsockname(s, reinterpret_cast<sockaddr*>(&addr), &len), 0);
  uint16_t port = ntohs(addr.sin_port);
  close(s);
  return port;
}

std::string httpGet(uint16_t port, std::string_view path) {
  int s = socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(s, 0);

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(port);
  if (connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
    close(s);
    return {};
  }

  const std::string request = "GET " + std::string(path) +
                              " HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                              "Connection: close\r\n\r\n";
  if (send(s, request.data(), request.size(), 0) !=
      static_cast<ssize_t>(request.size())) {
    close(s);
    return {};
  }

  std::string response;
  char buffer[4096];
  while (true) {
    const ssize_t bytes = recv(s, buffer, sizeof(buffer), 0);
    if (bytes <= 0) break;
    response.append(buffer, static_cast<size_t>(bytes));
  }
  close(s);
  return response;
}

std::string httpGetMetrics(uint16_t port) { return httpGet(port, "/metrics"); }

struct PrefillConnectionState {
  void setServerId(const std::string& serverId) {
    std::lock_guard<std::mutex> lock(mutex);
    this->serverId = serverId;
  }

  std::string getServerId() const {
    std::lock_guard<std::mutex> lock(mutex);
    return serverId;
  }

  mutable std::mutex mutex;
  std::string serverId;
};

// Mock prefill: SERVER mode, registers on connect, echoes a result per request.
class FakePrefill {
 public:
  FakePrefill(std::string serverId, uint16_t port, uint32_t maxInFlight = 4)
      : serverId_(std::move(serverId)), port_(port), maxInFlight_(maxInFlight) {
    sm_.initializeAsServer(port_);

    sm_.registerHandler<tt::sockets::RegistrationProbeMessage>(
        tt::sockets::tags::REGISTRATION_PROBE,
        [this](const tt::sockets::RegistrationProbeMessage&) {
          tt::sockets::PrefillRegistrationMessage msg;
          msg.server_id = serverId_;
          msg.max_in_flight = maxInFlight_;
          sm_.sendObject(tt::sockets::tags::PREFILL_REGISTRATION, msg);
        });

    sm_.registerHandler<tt::sockets::PrefillRequestMessage>(
        "prefill_request",
        [this](const tt::sockets::PrefillRequestMessage& req) {
          receivedTaskIds_.fetch_add(1);
          {
            std::lock_guard<std::mutex> lock(mutex_);
            lastRequest_ = req;
          }
          if (!autoReply_) return;
          tt::sockets::PrefillResultMessage res(req.task_id);
          res.error = false;
          res.finished = true;
          res.generated_text = "ok-from-" + serverId_;
          sm_.sendObject("prefill_result", res);
        });

    sm_.registerHandler<tt::sockets::CancelPrefillMessage>(
        tt::sockets::tags::CANCEL_PREFILL,
        [this](const tt::sockets::CancelPrefillMessage& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          cancelledTaskIds_.push_back(msg.task_id);
        });
  }

  ~FakePrefill() { sm_.stop(); }

  void start() { sm_.start(); }
  void stop() { sm_.stop(); }
  void setAutoReply(bool v) { autoReply_ = v; }
  uint32_t receivedTaskCount() const { return receivedTaskIds_.load(); }
  const std::string& serverId() const { return serverId_; }
  size_t cancelCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelledTaskIds_.size();
  }
  std::vector<uint32_t> cancelledTaskIds() {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelledTaskIds_;
  }

  std::optional<tt::sockets::PrefillRequestMessage> takeLastRequest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!lastRequest_) return std::nullopt;
    auto out = *lastRequest_;
    lastRequest_.reset();
    return out;
  }

 private:
  std::string serverId_;
  uint16_t port_;
  uint32_t maxInFlight_;
  std::atomic<bool> autoReply_{true};
  std::atomic<uint32_t> receivedTaskIds_{0};
  tt::sockets::SocketManager sm_;
  std::mutex mutex_;
  std::optional<tt::sockets::PrefillRequestMessage> lastRequest_;
  std::vector<uint32_t> cancelledTaskIds_;
};

// Mock decode: CLIENT to gateway, collects results/assignments.
class FakeDecode {
 public:
  FakeDecode(const std::string& gatewayHost, uint16_t gatewayPort) {
    sm_.initializeAsClient(gatewayHost, gatewayPort);
    sm_.setReconnectBackoff(std::chrono::milliseconds(50),
                            std::chrono::milliseconds(500));

    sm_.registerHandler<tt::sockets::PrefillResultMessage>(
        "prefill_result", [this](const tt::sockets::PrefillResultMessage& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          results_.push_back(msg);
          cv_.notify_all();
        });
    sm_.registerHandler<tt::sockets::PrefillAssignmentMessage>(
        tt::sockets::tags::PREFILL_ASSIGNMENT,
        [this](const tt::sockets::PrefillAssignmentMessage& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          assignments_.push_back(msg);
          cv_.notify_all();
        });
  }

  ~FakeDecode() { sm_.stop(); }

  void start() { sm_.start(); }
  void stop() { sm_.stop(); }

  void sendRequest(uint32_t taskId, size_t registrationHash = 0) {
    tt::sockets::PrefillRequestMessage req(taskId);
    if (registrationHash != 0)
      req.registration_hashes = {static_cast<uint64_t>(registrationHash)};
    sm_.sendObject("prefill_request", req);
  }

  void sendRequest(uint32_t taskId, std::vector<uint64_t> registrationHashes) {
    tt::sockets::PrefillRequestMessage req(taskId);
    req.registration_hashes = std::move(registrationHashes);
    sm_.sendObject("prefill_request", req);
  }

  void sendCancel(uint32_t taskId) {
    tt::sockets::CancelPrefillMessage cancel;
    cancel.task_id = taskId;
    sm_.sendObject(tt::sockets::tags::CANCEL_PREFILL, cancel);
  }

  bool isConnected() const { return sm_.isConnected(); }

  size_t resultCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return results_.size();
  }
  size_t assignmentCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return assignments_.size();
  }
  std::vector<tt::sockets::PrefillResultMessage> results() {
    std::lock_guard<std::mutex> lock(mutex_);
    return results_;
  }
  std::vector<tt::sockets::PrefillAssignmentMessage> assignments() {
    std::lock_guard<std::mutex> lock(mutex_);
    return assignments_;
  }

 private:
  tt::sockets::SocketManager sm_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<tt::sockets::PrefillResultMessage> results_;
  std::vector<tt::sockets::PrefillAssignmentMessage> assignments_;
};

// Real-sockets gateway wired the same way as main.cpp.
class GatewayHarness {
 public:
  GatewayHarness(uint16_t decodePort,
                 const std::vector<std::pair<std::string, uint16_t>>& prefills)
      : decodePort_(decodePort) {
    decodeSm_.initializeAsServer(decodePort_);

    for (const auto& [host, port] : prefills) {
      auto sm = std::make_unique<tt::sockets::SocketManager>();
      sm->setReconnectBackoff(std::chrono::milliseconds(50),
                              std::chrono::milliseconds(500));
      sm->initializeAsClient(host, port);
      prefillSms_.push_back(std::move(sm));
    }

    Dispatcher::Senders senders;
    senders.sendRequestToPrefill =
        [this](const std::string& serverId,
               const tt::sockets::PrefillRequestMessage& m) -> bool {
      auto* sm = registry_.getSocketManager(serverId);
      if (!sm) return false;
      return sm->sendObject("prefill_request", m);
    };
    senders.sendCancelToPrefill =
        [this](const std::string& serverId,
               const tt::sockets::CancelPrefillMessage& m) -> bool {
      auto* sm = registry_.getSocketManager(serverId);
      if (!sm) return false;
      return sm->sendObject(tt::sockets::tags::CANCEL_PREFILL, m);
    };
    senders.sendAssignmentToDecode =
        [this](const tt::sockets::PrefillAssignmentMessage& m) -> bool {
      return decodeSm_.sendObject(tt::sockets::tags::PREFILL_ASSIGNMENT, m);
    };
    senders.sendResultToDecode =
        [this](const tt::sockets::PrefillResultMessage& m) -> bool {
      return decodeSm_.sendObject("prefill_result", m);
    };

    dispatcher_ =
        std::make_unique<Dispatcher>(registry_, affinity_, std::move(senders));

    registry_.setOnPrefillDown(
        [this](const std::string& id) { dispatcher_->onPrefillDown(id); });

    for (auto& smPtr : prefillSms_) {
      auto* sm = smPtr.get();
      auto state = std::make_shared<PrefillConnectionState>();

      sm->registerHandler<tt::sockets::PrefillRegistrationMessage>(
          tt::sockets::tags::PREFILL_REGISTRATION,
          [this, sm,
           state](const tt::sockets::PrefillRegistrationMessage& msg) {
            state->setServerId(msg.server_id);
            registry_.preRegister(msg.server_id, sm);
            registry_.markRegistered(msg.server_id, msg.max_in_flight);
          });

      sm->registerHandler<tt::sockets::PrefillResultMessage>(
          "prefill_result",
          [this, state](const tt::sockets::PrefillResultMessage& msg) {
            dispatcher_->onPrefillResult(state->getServerId(), msg);
          });

      sm->setConnectionLostCallback([this, state] {
        const std::string sid = state->getServerId();
        if (!sid.empty()) registry_.markDown(sid);
      });
    }

    decodeSm_.registerHandler<tt::sockets::PrefillRequestMessage>(
        "prefill_request",
        [this](const tt::sockets::PrefillRequestMessage& msg) {
          dispatcher_->onPrefillRequest(msg);
        });

    decodeSm_.registerHandler<tt::sockets::CancelPrefillMessage>(
        tt::sockets::tags::CANCEL_PREFILL,
        [this](const tt::sockets::CancelPrefillMessage& msg) {
          dispatcher_->onPrefillCancel(msg);
        });
  }

  ~GatewayHarness() {
    proberStop_ = true;
    if (proberThread_.joinable()) proberThread_.join();
    decodeSm_.stop();
    for (auto& sm : prefillSms_) sm->stop();
  }

  void start() {
    for (auto& sm : prefillSms_) sm->start();
    decodeSm_.start();
    proberThread_ = std::thread([this] {
      while (!proberStop_.load()) {
        for (auto& sm : prefillSms_) {
          sm->sendObject(tt::sockets::tags::REGISTRATION_PROBE,
                         tt::sockets::RegistrationProbeMessage{});
        }
        std::this_thread::sleep_for(50ms);
      }
    });
  }

  PrefillRegistry& registry() { return registry_; }
  AffinityCache& affinity() { return affinity_; }
  Dispatcher& dispatcher() { return *dispatcher_; }

 private:
  uint16_t decodePort_;
  PrefillRegistry registry_;
  AffinityCache affinity_;
  tt::sockets::SocketManager decodeSm_;
  std::vector<std::unique_ptr<tt::sockets::SocketManager>> prefillSms_;
  std::unique_ptr<Dispatcher> dispatcher_;
  std::thread proberThread_;
  std::atomic<bool> proberStop_{false};
};

class FakeZmqPrefill {
 public:
  FakeZmqPrefill(std::string serverId, const std::string& routerHost,
                 uint16_t routerPort, uint32_t maxInFlight = 4)
      : serverId_(std::move(serverId)),
        routerHost_(routerHost),
        routerPort_(routerPort),
        maxInFlight_(maxInFlight) {
    sm_.initializeAsClient(routerHost_, routerPort_);
    sm_.setReconnectBackoff(std::chrono::milliseconds(50),
                            std::chrono::milliseconds(500));
  }

  ~FakeZmqPrefill() { stop(); }

  void start() {
    sm_.start();
    running_ = true;
    registrationThread_ = std::thread([this] {
      while (running_.load()) {
        tt::sockets::PrefillRegistrationMessage msg;
        msg.server_id = serverId_;
        msg.max_in_flight = maxInFlight_;
        sm_.sendRawData(tt::sockets::wire::serializeMessage(
            tt::sockets::tags::PREFILL_REGISTRATION, msg));
        std::this_thread::sleep_for(50ms);
      }
    });
    receiveThread_ = std::thread([this] {
      while (running_.load()) {
        auto data = sm_.receiveRawData();
        if (data.empty()) {
          std::this_thread::sleep_for(10ms);
          continue;
        }
        const std::string messageType =
            tt::sockets::wire::readMessageType(data);
        if (messageType == tt::sockets::tags::CANCEL_PREFILL) {
          auto cancel = tt::sockets::wire::deserializePayload<
              tt::sockets::CancelPrefillMessage>(data);
          {
            std::lock_guard<std::mutex> lock(mutex_);
            cancelledTaskIds_.push_back(cancel.task_id);
          }
          continue;
        }

        if (messageType != "prefill_request") {
          continue;
        }

        auto request = tt::sockets::wire::deserializePayload<
            tt::sockets::PrefillRequestMessage>(data);
        receivedTaskIds_.fetch_add(1);
        if (!autoReply_.load()) {
          continue;
        }
        tt::sockets::PrefillResultMessage result(request.task_id);
        result.finished = true;
        result.generated_text = "ok-from-" + serverId_;
        sm_.sendRawData(
            tt::sockets::wire::serializeMessage("prefill_result", result));
      }
    });
  }

  void stop() {
    running_ = false;
    if (registrationThread_.joinable()) {
      registrationThread_.join();
    }
    if (receiveThread_.joinable()) {
      receiveThread_.join();
    }
    sm_.stop();
  }

  uint32_t receivedTaskCount() const { return receivedTaskIds_.load(); }
  void setAutoReply(bool v) { autoReply_ = v; }
  size_t cancelCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelledTaskIds_.size();
  }
  std::vector<uint32_t> cancelledTaskIds() {
    std::lock_guard<std::mutex> lock(mutex_);
    return cancelledTaskIds_;
  }

 private:
  std::string serverId_;
  std::string routerHost_;
  uint16_t routerPort_;
  uint32_t maxInFlight_;
  std::atomic<bool> running_{false};
  std::atomic<bool> autoReply_{true};
  std::atomic<uint32_t> receivedTaskIds_{0};
  tt::sockets::ZmqSocketTransport sm_;
  std::mutex mutex_;
  std::vector<uint32_t> cancelledTaskIds_;
  std::thread registrationThread_;
  std::thread receiveThread_;
};

class ZmqRouterGatewayHarness {
 public:
  ZmqRouterGatewayHarness(uint16_t decodePort, uint16_t prefillRouterPort)
      : decodePort_(decodePort), prefillRouterPort_(prefillRouterPort) {
    decodeSm_.initializeAsServer(decodePort_);

    Dispatcher::Senders senders;
    senders.sendRequestToPrefill =
        [this](const std::string& serverId,
               const tt::sockets::PrefillRequestMessage& msg) -> bool {
      return prefillRouter_.sendObject(serverId, "prefill_request", msg);
    };
    senders.sendCancelToPrefill =
        [this](const std::string& serverId,
               const tt::sockets::CancelPrefillMessage& msg) -> bool {
      return prefillRouter_.sendObject(serverId,
                                       tt::sockets::tags::CANCEL_PREFILL, msg);
    };
    senders.sendAssignmentToDecode =
        [this](const tt::sockets::PrefillAssignmentMessage& msg) -> bool {
      return decodeSm_.sendObject(tt::sockets::tags::PREFILL_ASSIGNMENT, msg);
    };
    senders.sendResultToDecode =
        [this](const tt::sockets::PrefillResultMessage& msg) -> bool {
      return decodeSm_.sendObject("prefill_result", msg);
    };

    dispatcher_ =
        std::make_unique<Dispatcher>(registry_, affinity_, std::move(senders));

    registry_.setOnPrefillDown(
        [this](const std::string& id) { dispatcher_->onPrefillDown(id); });

    prefillRouter_.registerHandler<tt::sockets::PrefillRegistrationMessage>(
        tt::sockets::tags::PREFILL_REGISTRATION,
        [this](const ZmqPrefillRouter::PeerIdentity& peerId,
               const tt::sockets::PrefillRegistrationMessage& msg) {
          prefillRouter_.rememberRegistration(msg.server_id, peerId);
          registry_.preRegister(msg.server_id, nullptr);
          registry_.markRegistered(msg.server_id, msg.max_in_flight);
        });

    prefillRouter_.registerHandler<tt::sockets::PrefillResultMessage>(
        "prefill_result", [this](const ZmqPrefillRouter::PeerIdentity& peerId,
                                 const tt::sockets::PrefillResultMessage& msg) {
          auto serverId = prefillRouter_.serverIdForPeer(peerId);
          if (serverId.has_value()) {
            dispatcher_->onPrefillResult(*serverId, msg);
          }
        });

    decodeSm_.registerHandler<tt::sockets::PrefillRequestMessage>(
        "prefill_request",
        [this](const tt::sockets::PrefillRequestMessage& msg) {
          dispatcher_->onPrefillRequest(msg);
        });

    decodeSm_.registerHandler<tt::sockets::CancelPrefillMessage>(
        tt::sockets::tags::CANCEL_PREFILL,
        [this](const tt::sockets::CancelPrefillMessage& msg) {
          dispatcher_->onPrefillCancel(msg);
        });
  }

  ~ZmqRouterGatewayHarness() {
    decodeSm_.stop();
    prefillRouter_.stop();
  }

  void start() {
    ASSERT_TRUE(prefillRouter_.start("127.0.0.1", prefillRouterPort_));
    decodeSm_.start();
  }

  PrefillRegistry& registry() { return registry_; }
  AffinityCache& affinity() { return affinity_; }

 private:
  uint16_t decodePort_;
  uint16_t prefillRouterPort_;
  PrefillRegistry registry_;
  AffinityCache affinity_;
  tt::sockets::SocketManager decodeSm_;
  ZmqPrefillRouter prefillRouter_;
  std::unique_ptr<Dispatcher> dispatcher_;
};

class GatewayE2ETest : public ::testing::Test {
 protected:
  void SetUp() override {
    decodePort_ = ephemeralPort();
    prefillAPort_ = ephemeralPort();
    prefillBPort_ = ephemeralPort();

    prefillA_ = std::make_unique<FakePrefill>("prefill-A", prefillAPort_);
    prefillB_ = std::make_unique<FakePrefill>("prefill-B", prefillBPort_);
    prefillA_->start();
    prefillB_->start();

    gateway_ = std::make_unique<GatewayHarness>(
        decodePort_, std::vector<std::pair<std::string, uint16_t>>{
                         {"127.0.0.1", prefillAPort_},
                         {"127.0.0.1", prefillBPort_},
                     });
    gateway_->start();

    decode_ = std::make_unique<FakeDecode>("127.0.0.1", decodePort_);
    decode_->start();

    ASSERT_TRUE(waitFor([&] {
      auto snap = gateway_->registry().snapshot();
      int healthy = 0;
      for (const auto& s : snap)
        if (s.healthy) ++healthy;
      return healthy == 2 && decode_->isConnected();
    })) << "Timed out waiting for cluster to come up";
  }

  void TearDown() override {
    decode_.reset();
    gateway_.reset();
    prefillA_.reset();
    prefillB_.reset();
  }

  uint16_t decodePort_;
  uint16_t prefillAPort_;
  uint16_t prefillBPort_;
  std::unique_ptr<FakePrefill> prefillA_;
  std::unique_ptr<FakePrefill> prefillB_;
  std::unique_ptr<GatewayHarness> gateway_;
  std::unique_ptr<FakeDecode> decode_;
};

TEST_F(GatewayE2ETest, RequestIsRoutedAndResultFlowsBack) {
  decode_->sendRequest(/*task_id=*/1, /*hash=*/0);

  ASSERT_TRUE(waitFor([&] { return decode_->resultCount() >= 1; }))
      << "No result received within timeout";

  auto results = decode_->results();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].task_id, 1u);
  EXPECT_FALSE(results[0].error);
  EXPECT_TRUE(results[0].generated_text.rfind("ok-from-", 0) == 0);

  auto assignments = decode_->assignments();
  ASSERT_EQ(assignments.size(), 1u);
  EXPECT_EQ(assignments[0].task_id, 1u);
  EXPECT_TRUE(assignments[0].server_id == "prefill-A" ||
              assignments[0].server_id == "prefill-B");

  // Exactly one of the two prefills handled it.
  uint32_t total =
      prefillA_->receivedTaskCount() + prefillB_->receivedTaskCount();
  EXPECT_EQ(total, 1u);
}

TEST_F(GatewayE2ETest, RequestForwardsAllRegistrationHashesToPrefill) {
  const std::vector<uint64_t> hashes = {11, 22, 33};
  decode_->sendRequest(/*taskId=*/2, hashes);

  ASSERT_TRUE(waitFor([&] { return decode_->assignmentCount() >= 1; }));
  auto assignments = decode_->assignments();
  ASSERT_EQ(assignments.size(), 1u);
  ASSERT_TRUE(assignments[0].server_id == prefillA_->serverId() ||
              assignments[0].server_id == prefillB_->serverId());

  FakePrefill* assignedPrefill =
      assignments[0].server_id == prefillA_->serverId() ? prefillA_.get()
                                                        : prefillB_.get();
  ASSERT_TRUE(
      waitFor([&] { return assignedPrefill->receivedTaskCount() >= 1; }));

  auto request = assignedPrefill->takeLastRequest();
  ASSERT_TRUE(request.has_value());
  EXPECT_EQ(request->task_id, 2u);
  EXPECT_EQ(request->registration_hashes, hashes);
}

TEST_F(GatewayE2ETest, CancelIsForwardedToAssignedPrefill) {
  prefillA_->setAutoReply(false);
  prefillB_->setAutoReply(false);

  gateway_->affinity().record(/*hash=*/77, "prefill-A");
  decode_->sendRequest(/*task_id=*/88, /*hash=*/77);

  ASSERT_TRUE(waitFor([&] { return prefillA_->receivedTaskCount() >= 1; }));
  ASSERT_TRUE(waitFor([&] { return decode_->assignmentCount() >= 1; }));

  decode_->sendCancel(/*taskId=*/88);

  ASSERT_TRUE(waitFor([&] { return prefillA_->cancelCount() >= 1; }))
      << "Prefill should receive cancellation routed through gateway";
  auto cancelled = prefillA_->cancelledTaskIds();
  ASSERT_EQ(cancelled.size(), 1u);
  EXPECT_EQ(cancelled[0], 88u);
  EXPECT_EQ(prefillB_->cancelCount(), 0u);
  EXPECT_EQ(decode_->resultCount(), 0u);
}

TEST_F(GatewayE2ETest, RequestTimeoutFailsTaskToDecode) {
  prefillA_->setAutoReply(false);
  prefillB_->setAutoReply(false);

  gateway_->affinity().record(/*hash=*/77, "prefill-A");
  decode_->sendRequest(/*task_id=*/89, /*hash=*/77);

  ASSERT_TRUE(waitFor([&] { return prefillA_->receivedTaskCount() >= 1; }));
  ASSERT_TRUE(waitFor([&] { return decode_->assignmentCount() >= 1; }));
  EXPECT_EQ(decode_->resultCount(), 0u);

  gateway_->dispatcher().onRequestTimeouts(Dispatcher::Clock::now() +
                                           std::chrono::minutes(6));

  ASSERT_TRUE(waitFor([&] { return decode_->resultCount() >= 1; }));
  ASSERT_TRUE(waitFor([&] { return prefillA_->cancelCount() >= 1; }));
  auto results = decode_->results();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].task_id, 89u);
  EXPECT_TRUE(results[0].error);
  EXPECT_TRUE(results[0].finished);
  EXPECT_EQ(results[0].generated_text, "timeout");

  auto cancelled = prefillA_->cancelledTaskIds();
  ASSERT_EQ(cancelled.size(), 1u);
  EXPECT_EQ(cancelled[0], 89u);
}

TEST_F(GatewayE2ETest, StickyRoutingByRegistrationHash) {
  decode_->sendRequest(/*task_id=*/1, /*hash=*/42);
  ASSERT_TRUE(waitFor([&] { return decode_->resultCount() >= 1; }));
  auto firstAssignments = decode_->assignments();
  ASSERT_EQ(firstAssignments.size(), 1u);
  const std::string firstServer = firstAssignments[0].server_id;

  decode_->sendRequest(/*task_id=*/2, /*hash=*/42);
  ASSERT_TRUE(waitFor([&] { return decode_->resultCount() >= 2; }));
  auto allAssignments = decode_->assignments();
  ASSERT_EQ(allAssignments.size(), 2u);
  EXPECT_EQ(allAssignments[1].server_id, firstServer)
      << "Sticky routing should reuse the previous prefill";
  EXPECT_EQ(allAssignments[1].task_id, 2u);
}

TEST_F(GatewayE2ETest, PrefillDownFailsInFlightTaskToDecode) {
  prefillA_->setAutoReply(false);
  prefillB_->setAutoReply(false);

  // Seed affinity so we know which prefill will take the request.
  gateway_->affinity().record(/*hash=*/77, "prefill-A");
  decode_->sendRequest(/*task_id=*/55, /*hash=*/77);

  ASSERT_TRUE(waitFor([&] { return prefillA_->receivedTaskCount() >= 1; }));
  ASSERT_TRUE(waitFor([&] { return decode_->assignmentCount() >= 1; }));
  EXPECT_EQ(decode_->resultCount(), 0u);

  prefillA_->stop();

  ASSERT_TRUE(
      waitFor([&] { return decode_->resultCount() >= 1; }, /*timeout=*/3s))
      << "Decode should receive a failure result when prefill drops";

  auto results = decode_->results();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].task_id, 55u);
  EXPECT_TRUE(results[0].error);
  EXPECT_EQ(results[0].generated_text, "prefill_down");
  EXPECT_FALSE(gateway_->affinity().lookup(77).has_value());
}

TEST(ZmqRouterGatewayE2ETest, PrefillsCanStartBeforeGateway) {
  const uint16_t decodePort = ephemeralPort();
  const uint16_t prefillRouterPort = ephemeralPort();

  FakeZmqPrefill prefillA("prefill-A", "127.0.0.1", prefillRouterPort);
  FakeZmqPrefill prefillB("prefill-B", "127.0.0.1", prefillRouterPort);
  prefillA.start();
  prefillB.start();

  ZmqRouterGatewayHarness gateway(decodePort, prefillRouterPort);
  gateway.start();

  FakeDecode decode("127.0.0.1", decodePort);
  decode.start();

  ASSERT_TRUE(waitFor([&] {
    auto snap = gateway.registry().snapshot();
    int healthy = 0;
    for (const auto& s : snap) {
      if (s.healthy) ++healthy;
    }
    return healthy == 2 && decode.isConnected();
  })) << "Timed out waiting for prefills to register after gateway start";

  decode.sendRequest(/*task_id=*/101, /*hash=*/0);
  decode.sendRequest(/*task_id=*/102, /*hash=*/0);

  ASSERT_TRUE(waitFor([&] { return decode.resultCount() >= 2; }))
      << "No results received through ZMQ prefill ROUTER";

  auto assignments = decode.assignments();
  ASSERT_EQ(assignments.size(), 2u);
  EXPECT_NE(assignments[0].server_id, assignments[1].server_id);

  EXPECT_EQ(prefillA.receivedTaskCount() + prefillB.receivedTaskCount(), 2u);
  EXPECT_EQ(prefillA.receivedTaskCount(), 1u);
  EXPECT_EQ(prefillB.receivedTaskCount(), 1u);
}

TEST(ZmqRouterGatewayE2ETest, CancelIsForwardedToAssignedPrefill) {
  const uint16_t decodePort = ephemeralPort();
  const uint16_t prefillRouterPort = ephemeralPort();

  ZmqRouterGatewayHarness gateway(decodePort, prefillRouterPort);
  gateway.start();

  FakeZmqPrefill prefillA("prefill-A", "127.0.0.1", prefillRouterPort);
  FakeZmqPrefill prefillB("prefill-B", "127.0.0.1", prefillRouterPort);
  prefillA.setAutoReply(false);
  prefillB.setAutoReply(false);
  prefillA.start();
  prefillB.start();

  FakeDecode decode("127.0.0.1", decodePort);
  decode.start();

  ASSERT_TRUE(waitFor([&] {
    auto snap = gateway.registry().snapshot();
    int healthy = 0;
    for (const auto& s : snap) {
      if (s.healthy) ++healthy;
    }
    return healthy == 2 && decode.isConnected();
  })) << "Timed out waiting for ZMQ gateway cluster";

  gateway.affinity().record(/*hash=*/77, "prefill-A");
  decode.sendRequest(/*task_id=*/303, /*hash=*/77);

  ASSERT_TRUE(waitFor([&] { return prefillA.receivedTaskCount() >= 1; }));
  ASSERT_TRUE(waitFor([&] { return decode.assignmentCount() >= 1; }));

  decode.sendCancel(/*taskId=*/303);

  ASSERT_TRUE(waitFor([&] { return prefillA.cancelCount() >= 1; }))
      << "ZMQ prefill should receive cancellation routed through gateway";
  auto cancelled = prefillA.cancelledTaskIds();
  ASSERT_EQ(cancelled.size(), 1u);
  EXPECT_EQ(cancelled[0], 303u);
  EXPECT_EQ(prefillB.cancelCount(), 0u);
  EXPECT_EQ(decode.resultCount(), 0u);
}

TEST(ZmqPrefillRouterTest, RoutesRequestToRegisteredPrefill) {
  const uint16_t port = ephemeralPort();

  ZmqPrefillRouter router;
  std::mutex mutex;
  std::condition_variable cv;
  bool registered = false;
  bool requestReceived = false;
  bool resultReceived = false;

  router.registerHandler<tt::sockets::PrefillRegistrationMessage>(
      tt::sockets::tags::PREFILL_REGISTRATION,
      [&router, &mutex, &cv, &registered](
          const ZmqPrefillRouter::PeerIdentity& peerId,
          const tt::sockets::PrefillRegistrationMessage& msg) {
        router.rememberRegistration(msg.server_id, peerId);
        {
          std::lock_guard<std::mutex> lock(mutex);
          registered = true;
        }
        cv.notify_all();
      });

  router.registerHandler<tt::sockets::PrefillResultMessage>(
      "prefill_result", [&mutex, &cv, &resultReceived](
                            const ZmqPrefillRouter::PeerIdentity&,
                            const tt::sockets::PrefillResultMessage& msg) {
        EXPECT_EQ(msg.task_id, 7u);
        std::lock_guard<std::mutex> lock(mutex);
        resultReceived = true;
        cv.notify_all();
      });

  ASSERT_TRUE(router.start("127.0.0.1", port));

  tt::sockets::ZmqSocketTransport client;
  ASSERT_TRUE(client.initializeAsClient("127.0.0.1", port));
  client.start();

  std::atomic<bool> clientRunning{true};
  std::thread clientThread([&] {
    while (clientRunning.load()) {
      auto data = client.receiveRawData();
      if (data.empty()) {
        std::this_thread::sleep_for(10ms);
        continue;
      }

      if (tt::sockets::wire::readMessageType(data) != "prefill_request") {
        continue;
      }

      auto request = tt::sockets::wire::deserializePayload<
          tt::sockets::PrefillRequestMessage>(data);
      EXPECT_EQ(request.task_id, 7u);
      {
        std::lock_guard<std::mutex> lock(mutex);
        requestReceived = true;
      }
      cv.notify_all();

      tt::sockets::PrefillResultMessage result(request.task_id);
      result.finished = true;
      result.generated_text = "ok";
      client.sendRawData(
          tt::sockets::wire::serializeMessage("prefill_result", result));
    }
  });

  tt::sockets::PrefillRegistrationMessage registration;
  registration.server_id = "prefill-A";
  registration.max_in_flight = 4;
  ASSERT_TRUE(client.sendRawData(tt::sockets::wire::serializeMessage(
      tt::sockets::tags::PREFILL_REGISTRATION, registration)));
  ASSERT_TRUE(waitFor([&] {
    std::lock_guard<std::mutex> lock(mutex);
    return registered;
  }));

  tt::sockets::PrefillRequestMessage request(7);
  ASSERT_TRUE(router.sendObject("prefill-A", "prefill_request", request));
  ASSERT_TRUE(waitFor([&] {
    std::lock_guard<std::mutex> lock(mutex);
    return requestReceived && resultReceived;
  }));

  clientRunning = false;
  if (clientThread.joinable()) {
    clientThread.join();
  }
  client.stop();
  router.stop();
}

TEST(ZmqSocketOptionsTest, AppliesHeartbeatsAndMandatoryRouterSends) {
  zmq::context_t context(tt::sockets::zmq_options::CONTEXT_IO_THREADS);

  zmq::socket_t dealer(context, zmq::socket_type::dealer);
  ASSERT_NO_THROW(tt::sockets::zmq_options::applyCommonOptions(
      dealer, /*receiveTimeoutMs=*/42));
  EXPECT_EQ(dealer.get(zmq::sockopt::linger), 0);
  EXPECT_EQ(dealer.get(zmq::sockopt::rcvtimeo), 42);
  EXPECT_EQ(dealer.get(zmq::sockopt::heartbeat_ivl),
            tt::sockets::zmq_options::HEARTBEAT_INTERVAL_MS);
  EXPECT_EQ(dealer.get(zmq::sockopt::heartbeat_timeout),
            tt::sockets::zmq_options::HEARTBEAT_TIMEOUT_MS);

  zmq::socket_t router(context, zmq::socket_type::router);
  ASSERT_NO_THROW(tt::sockets::zmq_options::applyRouterOptions(
      router, /*receiveTimeoutMs=*/43));
  EXPECT_EQ(router.get(zmq::sockopt::rcvtimeo), 43);

  dealer.close();
  router.close();
  context.close();
}

TEST(ZmqPrefillRouterTest, SendFailsWhenRegisteredPeerIsNoLongerRoutable) {
  const uint16_t port = ephemeralPort();

  ZmqPrefillRouter router;
  std::mutex mutex;
  bool registered = false;

  router.registerHandler<tt::sockets::PrefillRegistrationMessage>(
      tt::sockets::tags::PREFILL_REGISTRATION,
      [&router, &mutex, &registered](
          const ZmqPrefillRouter::PeerIdentity& peerId,
          const tt::sockets::PrefillRegistrationMessage& msg) {
        router.rememberRegistration(msg.server_id, peerId);
        std::lock_guard<std::mutex> lock(mutex);
        registered = true;
      });

  ASSERT_TRUE(router.start("127.0.0.1", port));

  tt::sockets::ZmqSocketTransport client;
  ASSERT_TRUE(client.initializeAsClient("127.0.0.1", port));
  client.start();

  tt::sockets::PrefillRegistrationMessage registration;
  registration.server_id = "prefill-A";
  registration.max_in_flight = 4;
  ASSERT_TRUE(client.sendRawData(tt::sockets::wire::serializeMessage(
      tt::sockets::tags::PREFILL_REGISTRATION, registration)));
  ASSERT_TRUE(waitFor([&] {
    std::lock_guard<std::mutex> lock(mutex);
    return registered;
  }));

  client.stop();

  tt::sockets::PrefillRequestMessage request(7);
  ASSERT_TRUE(waitFor(
      [&] {
        return !router.sendObject("prefill-A", "prefill_request", request);
      },
      /*timeout=*/3s))
      << "ROUTER_MANDATORY should make sends to unroutable peers fail";

  router.stop();
}

TEST(GatewayMetricsServerTest, ServesPrometheusTextOnMetricsPath) {
  auto& metrics = GatewayMetrics::instance();
  metrics.resetForTests();
  metrics.recordRoutingDecision("least_inflight");

  const uint16_t port = ephemeralPort();
  GatewayMetricsServer server(metrics);
  ASSERT_TRUE(server.start(port));

  ASSERT_TRUE(waitFor([&] {
    const std::string response = httpGetMetrics(port);
    return response.find("HTTP/1.1 200 OK") != std::string::npos &&
           response.find("tt_gateway_routing_decisions_total") !=
               std::string::npos;
  }));

  server.stop();
}

TEST(GatewayMetricsServerTest, ServesHealthJsonOnLivenessPath) {
  auto& metrics = GatewayMetrics::instance();
  metrics.resetForTests();

  const uint16_t port = ephemeralPort();
  GatewayMetricsServer server(metrics);
  server.setHealthProvider([] {
    return std::string(
        R"({"status":"alive","transport":"tcp","registered_prefills":2,"healthy_prefills":2,"accepting_prefills":2,"decode_connected":false})"
        "\n");
  });
  ASSERT_TRUE(server.start(port));

  ASSERT_TRUE(waitFor([&] {
    const std::string response = httpGet(port, "/tt-liveness");
    return response.find("HTTP/1.1 200 OK") != std::string::npos &&
           response.find("Content-Type: application/json") !=
               std::string::npos &&
           response.find(R"("healthy_prefills":2)") != std::string::npos &&
           response.find(R"("decode_connected":false)") != std::string::npos;
  }));

  server.stop();
}

}  // namespace
}  // namespace tt::gateway
