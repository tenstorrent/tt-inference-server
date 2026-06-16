// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// End-to-end integration test for PrefillGateway over real loopback sockets.
// Validates registration handshake, routing (round-robin + prefix match),
// result delivery, and prefill-down failover.

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

#include "gateway/dispatcher.hpp"
#include "gateway/gateway_health.hpp"
#include "gateway/gateway_health_server.hpp"
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

uint64_t fakeMigrationId(uint32_t taskId) { return 1000ULL + taskId; }

std::vector<int64_t> fakeTokenIds(uint32_t taskId) {
  return {static_cast<int64_t>(taskId * 10 + 1),
          static_cast<int64_t>(taskId * 10 + 2)};
}

void populateFakePrefillResult(tt::sockets::PrefillResultMessage& result,
                               const std::string& serverId) {
  result.error = false;
  result.generatedText = "ok-from-" + serverId;
  result.tokenIds = fakeTokenIds(result.taskId);
  result.remainingTokens = 3;
  result.slotId = 42u;
  result.temperature = 0.7f;
  result.topP = 0.9f;
  result.topK = 11;
  result.fastMode = true;
  result.cachedTokens = 5;
  result.migrationId = fakeMigrationId(result.taskId);
}

void expectFakePrefillResultPayload(
    const tt::sockets::PrefillResultMessage& result) {
  EXPECT_FALSE(result.error);
  EXPECT_EQ(result.tokenIds, fakeTokenIds(result.taskId));
  EXPECT_EQ(result.remainingTokens, 3);
  EXPECT_EQ(result.slotId, 42u);
  EXPECT_EQ(result.temperature, 0.7f);
  EXPECT_EQ(result.topP, 0.9f);
  EXPECT_EQ(result.topK, 11);
  EXPECT_TRUE(result.fastMode);
  EXPECT_EQ(result.cachedTokens, 5);
  EXPECT_EQ(result.migrationId, fakeMigrationId(result.taskId));
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
          msg.serverId = serverId_;
          msg.maxInFlight = maxInFlight_;
          sm_.sendObject(tt::sockets::tags::PREFILL_REGISTRATION, msg);
        });

    sm_.registerHandler<tt::sockets::PrefillRequestMessage>(
        tt::sockets::tags::PREFILL_REQUEST,
        [this](const tt::sockets::PrefillRequestMessage& req) {
          receivedTaskIds_.fetch_add(1);
          {
            std::lock_guard<std::mutex> lock(mutex_);
            lastRequest_ = req;
          }
          if (!autoReply_) return;
          tt::sockets::PrefillResultMessage res(req.taskId);
          populateFakePrefillResult(res, serverId_);
          sm_.sendObject(tt::sockets::tags::PREFILL_RESULT, res);
        });

    sm_.registerHandler<tt::sockets::CancelPrefillMessage>(
        tt::sockets::tags::CANCEL_PREFILL,
        [this](const tt::sockets::CancelPrefillMessage& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          cancelledTaskIds_.push_back(msg.taskId);
        });
  }

  ~FakePrefill() { sm_.stop(); }

  void start() { sm_.start(); }
  void stop() { sm_.stop(); }
  void setAutoReply(bool v) { autoReply_ = v; }
  void sendCacheBlocksAdded(std::vector<uint64_t> blockHashes) {
    tt::sockets::PrefillCacheBlocksAddedMessage msg;
    msg.serverId = serverId_;
    msg.blockHashes = std::move(blockHashes);
    sm_.sendObject(tt::sockets::tags::PREFILL_CACHE_BLOCKS_ADDED, msg);
  }
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

// Mock decode: CLIENT to gateway, collects results.
class FakeDecode {
 public:
  FakeDecode(const std::string& gatewayHost, uint16_t gatewayPort) {
    sm_.initializeAsClient(gatewayHost, gatewayPort);
    sm_.setReconnectBackoff(std::chrono::milliseconds(50),
                            std::chrono::milliseconds(500));

    sm_.registerHandler<tt::sockets::PrefillResultMessage>(
        tt::sockets::tags::PREFILL_RESULT,
        [this](const tt::sockets::PrefillResultMessage& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          results_.push_back(msg);
          cv_.notify_all();
        });
    sm_.registerHandler<tt::sockets::PrefillHealthStatusMessage>(
        tt::sockets::tags::PREFILL_HEALTH_STATUS,
        [this](const tt::sockets::PrefillHealthStatusMessage& msg) {
          std::lock_guard<std::mutex> lock(mutex_);
          healthStatuses_.push_back(msg);
          cv_.notify_all();
        });
  }

  ~FakeDecode() { sm_.stop(); }

  void start() { sm_.start(); }
  void stop() { sm_.stop(); }

  void sendRequest(uint32_t taskId, size_t registrationHash = 0) {
    tt::sockets::PrefillRequestMessage req(taskId);
    if (registrationHash != 0)
      req.registrationHashes = {static_cast<uint64_t>(registrationHash)};
    sm_.sendObject(tt::sockets::tags::PREFILL_REQUEST, req);
  }

  void sendRequest(uint32_t taskId, std::vector<uint64_t> registrationHashes) {
    tt::sockets::PrefillRequestMessage req(taskId);
    req.registrationHashes = std::move(registrationHashes);
    sm_.sendObject(tt::sockets::tags::PREFILL_REQUEST, req);
  }

  void sendRequest(tt::sockets::PrefillRequestMessage req) {
    sm_.sendObject(tt::sockets::tags::PREFILL_REQUEST, req);
  }

  void sendCancel(uint32_t taskId) {
    tt::sockets::CancelPrefillMessage cancel;
    cancel.taskId = taskId;
    sm_.sendObject(tt::sockets::tags::CANCEL_PREFILL, cancel);
  }

  void sendHealthRequest() {
    sm_.sendObject(tt::sockets::tags::PREFILL_HEALTH_REQUEST,
                   tt::sockets::PrefillHealthRequestMessage{});
  }

  bool isConnected() const { return sm_.isConnected(); }

  size_t resultCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return results_.size();
  }
  std::vector<tt::sockets::PrefillResultMessage> results() {
    std::lock_guard<std::mutex> lock(mutex_);
    return results_;
  }
  std::vector<tt::sockets::PrefillHealthStatusMessage> healthStatuses() {
    std::lock_guard<std::mutex> lock(mutex_);
    return healthStatuses_;
  }

 private:
  tt::sockets::SocketManager sm_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<tt::sockets::PrefillResultMessage> results_;
  std::vector<tt::sockets::PrefillHealthStatusMessage> healthStatuses_;
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
      return sm->sendObject(tt::sockets::tags::PREFILL_REQUEST, m);
    };
    senders.sendCancelToPrefill =
        [this](const std::string& serverId,
               const tt::sockets::CancelPrefillMessage& m) -> bool {
      auto* sm = registry_.getSocketManager(serverId);
      if (!sm) return false;
      return sm->sendObject(tt::sockets::tags::CANCEL_PREFILL, m);
    };
    senders.sendResultToDecode =
        [this](const tt::sockets::PrefillResultMessage& m) -> bool {
      return decodeSm_.sendObject(tt::sockets::tags::PREFILL_RESULT, m);
    };

    dispatcher_ = std::make_unique<Dispatcher>(registry_, std::move(senders));

    registry_.setOnPrefillDown(
        [this](const std::string& id) { dispatcher_->onPrefillDown(id); });

    for (auto& smPtr : prefillSms_) {
      auto* sm = smPtr.get();
      auto state = std::make_shared<PrefillConnectionState>();

      sm->registerHandler<tt::sockets::PrefillRegistrationMessage>(
          tt::sockets::tags::PREFILL_REGISTRATION,
          [this, sm,
           state](const tt::sockets::PrefillRegistrationMessage& msg) {
            state->setServerId(msg.serverId);
            registry_.preRegister(msg.serverId, sm);
            registry_.markRegistered(msg.serverId, msg.maxInFlight);
          });

      sm->registerHandler<tt::sockets::PrefillResultMessage>(
          tt::sockets::tags::PREFILL_RESULT,
          [this, state](const tt::sockets::PrefillResultMessage& msg) {
            dispatcher_->onPrefillResult(state->getServerId(), msg);
          });

      sm->registerHandler<tt::sockets::PrefillCacheBlocksAddedMessage>(
          tt::sockets::tags::PREFILL_CACHE_BLOCKS_ADDED,
          [this](const tt::sockets::PrefillCacheBlocksAddedMessage& msg) {
            dispatcher_->onCacheBlocksAdded(msg);
          });

      sm->setConnectionLostCallback([this, state] {
        const std::string sid = state->getServerId();
        if (!sid.empty()) registry_.markDown(sid);
      });
    }

    decodeSm_.registerHandler<tt::sockets::PrefillRequestMessage>(
        tt::sockets::tags::PREFILL_REQUEST,
        [this](const tt::sockets::PrefillRequestMessage& msg) {
          dispatcher_->onPrefillRequest(msg);
        });

    decodeSm_.registerHandler<tt::sockets::CancelPrefillMessage>(
        tt::sockets::tags::CANCEL_PREFILL,
        [this](const tt::sockets::CancelPrefillMessage& msg) {
          dispatcher_->onPrefillCancel(msg);
        });
    decodeSm_.registerHandler<tt::sockets::PrefillHealthRequestMessage>(
        tt::sockets::tags::PREFILL_HEALTH_REQUEST,
        [this](const tt::sockets::PrefillHealthRequestMessage&) {
          const auto health = buildGatewayHealthStatus(registry_, "tcp",
                                                       decodeSm_.isConnected());
          tt::sockets::PrefillHealthStatusMessage response;
          response.ready = health.ready;
          decodeSm_.sendObject(tt::sockets::tags::PREFILL_HEALTH_STATUS,
                               response);
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
  Dispatcher& dispatcher() { return *dispatcher_; }

 private:
  uint16_t decodePort_;
  PrefillRegistry registry_;
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
        msg.serverId = serverId_;
        msg.maxInFlight = maxInFlight_;
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
            cancelledTaskIds_.push_back(cancel.taskId);
          }
          continue;
        }

        if (messageType != tt::sockets::tags::PREFILL_REQUEST) {
          continue;
        }

        auto request = tt::sockets::wire::deserializePayload<
            tt::sockets::PrefillRequestMessage>(data);
        receivedTaskIds_.fetch_add(1);
        if (!autoReply_.load()) {
          continue;
        }
        tt::sockets::PrefillResultMessage result(request.taskId);
        populateFakePrefillResult(result, serverId_);
        sm_.sendRawData(tt::sockets::wire::serializeMessage(
            tt::sockets::tags::PREFILL_RESULT, result));
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
      return prefillRouter_.sendObject(serverId,
                                       tt::sockets::tags::PREFILL_REQUEST, msg);
    };
    senders.sendCancelToPrefill =
        [this](const std::string& serverId,
               const tt::sockets::CancelPrefillMessage& msg) -> bool {
      return prefillRouter_.sendObject(serverId,
                                       tt::sockets::tags::CANCEL_PREFILL, msg);
    };
    senders.sendResultToDecode =
        [this](const tt::sockets::PrefillResultMessage& msg) -> bool {
      return decodeSm_.sendObject(tt::sockets::tags::PREFILL_RESULT, msg);
    };

    dispatcher_ = std::make_unique<Dispatcher>(registry_, std::move(senders));

    registry_.setOnPrefillDown(
        [this](const std::string& id) { dispatcher_->onPrefillDown(id); });

    prefillRouter_.registerHandler<tt::sockets::PrefillRegistrationMessage>(
        tt::sockets::tags::PREFILL_REGISTRATION,
        [this](const ZmqPrefillRouter::PeerIdentity& peerId,
               const tt::sockets::PrefillRegistrationMessage& msg) {
          prefillRouter_.rememberRegistration(msg.serverId, peerId);
          registry_.preRegister(msg.serverId, nullptr);
          registry_.markRegistered(msg.serverId, msg.maxInFlight);
        });

    prefillRouter_.registerHandler<tt::sockets::PrefillResultMessage>(
        tt::sockets::tags::PREFILL_RESULT,
        [this](const ZmqPrefillRouter::PeerIdentity& peerId,
               const tt::sockets::PrefillResultMessage& msg) {
          auto serverId = prefillRouter_.serverIdForPeer(peerId);
          if (serverId.has_value()) {
            dispatcher_->onPrefillResult(*serverId, msg);
          }
        });

    decodeSm_.registerHandler<tt::sockets::PrefillRequestMessage>(
        tt::sockets::tags::PREFILL_REQUEST,
        [this](const tt::sockets::PrefillRequestMessage& msg) {
          dispatcher_->onPrefillRequest(msg);
        });

    decodeSm_.registerHandler<tt::sockets::CancelPrefillMessage>(
        tt::sockets::tags::CANCEL_PREFILL,
        [this](const tt::sockets::CancelPrefillMessage& msg) {
          dispatcher_->onPrefillCancel(msg);
        });
    decodeSm_.registerHandler<tt::sockets::PrefillHealthRequestMessage>(
        tt::sockets::tags::PREFILL_HEALTH_REQUEST,
        [this](const tt::sockets::PrefillHealthRequestMessage&) {
          const auto health = buildGatewayHealthStatus(registry_, "zmq",
                                                       decodeSm_.isConnected());
          tt::sockets::PrefillHealthStatusMessage response;
          response.ready = health.ready;
          decodeSm_.sendObject(tt::sockets::tags::PREFILL_HEALTH_STATUS,
                               response);
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

 private:
  uint16_t decodePort_;
  uint16_t prefillRouterPort_;
  PrefillRegistry registry_;
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
  EXPECT_EQ(results[0].taskId, 1u);
  EXPECT_TRUE(results[0].generatedText.rfind("ok-from-", 0) == 0);
  expectFakePrefillResultPayload(results[0]);

  // Exactly one of the two prefills handled it.
  uint32_t total =
      prefillA_->receivedTaskCount() + prefillB_->receivedTaskCount();
  EXPECT_EQ(total, 1u);
}

TEST_F(GatewayE2ETest, HealthProbeReportsReadyPrefills) {
  decode_->sendHealthRequest();

  ASSERT_TRUE(waitFor([&] { return !decode_->healthStatuses().empty(); }))
      << "Decode should receive gateway prefill health status";
  const auto statuses = decode_->healthStatuses();
  ASSERT_EQ(statuses.size(), 1u);
  EXPECT_TRUE(statuses[0].ready);
}

TEST_F(GatewayE2ETest, RequestForwardsPrefillPayloadToPrefill) {
  tt::sockets::PrefillRequestMessage sent(2);
  sent.registrationHashes = {11, 22, 33};
  sent.tokenIds = {101, 102, 103};
  sent.maxTokens = 7;
  sent.slotId = 42u;
  sent.temperature = 0.7f;
  sent.topP = 0.9f;
  sent.topK = 11;
  sent.fastMode = true;
  sent.decodePositionId = 12;
  sent.decodeSkipTokens = 10;

  decode_->sendRequest(sent);

  ASSERT_TRUE(waitFor([&] {
    return prefillA_->receivedTaskCount() + prefillB_->receivedTaskCount() >= 1;
  }));

  FakePrefill* assignedPrefill =
      prefillA_->receivedTaskCount() > 0 ? prefillA_.get() : prefillB_.get();

  auto request = assignedPrefill->takeLastRequest();
  ASSERT_TRUE(request.has_value());
  EXPECT_EQ(request->task_id, sent.taskId);
  EXPECT_EQ(request->registration_hashes, sent.registrationHashes);
  EXPECT_EQ(request->token_ids, sent.tokenIds);
  EXPECT_EQ(request->max_tokens, sent.maxTokens);
  EXPECT_EQ(request->slot_id, sent.slotId);
  EXPECT_EQ(request->temperature, sent.temperature);
  EXPECT_EQ(request->top_p, sent.topP);
  EXPECT_EQ(request->top_k, sent.topK);
  EXPECT_EQ(request->fast_mode, sent.fastMode);
  EXPECT_EQ(request->decode_position_id, sent.decodePositionId);
  EXPECT_EQ(request->decode_skip_tokens, sent.decodeSkipTokens);
}

TEST_F(GatewayE2ETest, CancelIsForwardedToAssignedPrefill) {
  prefillA_->setAutoReply(false);
  prefillB_->setAutoReply(false);

  gateway_->registry().addCachedBlocks("prefill-A", {77});
  decode_->sendRequest(/*task_id=*/88, /*hash=*/77);

  ASSERT_TRUE(waitFor([&] { return prefillA_->receivedTaskCount() >= 1; }));

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

  gateway_->registry().addCachedBlocks("prefill-A", {77});
  decode_->sendRequest(/*task_id=*/89, /*hash=*/77);

  ASSERT_TRUE(waitFor([&] { return prefillA_->receivedTaskCount() >= 1; }));
  EXPECT_EQ(decode_->resultCount(), 0u);

  gateway_->dispatcher().onRequestTimeouts(Dispatcher::Clock::now() +
                                           std::chrono::minutes(6));

  ASSERT_TRUE(waitFor([&] { return decode_->resultCount() >= 1; }));
  ASSERT_TRUE(waitFor([&] { return prefillA_->cancelCount() >= 1; }));
  auto results = decode_->results();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].taskId, 89u);
  EXPECT_TRUE(results[0].error);
  EXPECT_EQ(results[0].generatedText, "timeout");

  auto cancelled = prefillA_->cancelledTaskIds();
  ASSERT_EQ(cancelled.size(), 1u);
  EXPECT_EQ(cancelled[0], 89u);
}

TEST_F(GatewayE2ETest, PrefixRoutingByRegistrationHash) {
  gateway_->registry().addCachedBlocks("prefill-A", {42});

  decode_->sendRequest(/*task_id=*/2, /*hash=*/42);
  ASSERT_TRUE(waitFor([&] { return decode_->resultCount() >= 1; }));
  EXPECT_EQ(prefillA_->receivedTaskCount(), 1u)
      << "Prefix routing should choose the prefill with the cached block";
  EXPECT_EQ(prefillB_->receivedTaskCount(), 0u);
}

TEST_F(GatewayE2ETest, CacheNotificationDrivesPrefixRouting) {
  const std::vector<uint64_t> cachedBlocks = {42, 43, 44};
  prefillB_->sendCacheBlocksAdded(cachedBlocks);

  ASSERT_TRUE(waitFor([&] {
    for (const auto& snapshot : gateway_->registry().snapshot()) {
      if (snapshot.serverId == prefillB_->serverId()) {
        return snapshot.cachedBlocks == cachedBlocks.size();
      }
    }
    return false;
  })) << "Gateway should learn prefill-B cache blocks from socket notification";

  decode_->sendRequest(/*taskId=*/3, {42, 43, 44, 99});

  ASSERT_TRUE(waitFor([&] { return decode_->resultCount() >= 1; }));
  EXPECT_EQ(prefillB_->receivedTaskCount(), 1u);
  EXPECT_EQ(prefillA_->receivedTaskCount(), 0u);
}

TEST_F(GatewayE2ETest, PrefillDownFailsInFlightTaskToDecode) {
  prefillA_->setAutoReply(false);
  prefillB_->setAutoReply(false);

  // Seed the cache view so we know which prefill will take the request.
  gateway_->registry().addCachedBlocks("prefill-A", {77});
  decode_->sendRequest(/*task_id=*/55, /*hash=*/77);

  ASSERT_TRUE(waitFor([&] { return prefillA_->receivedTaskCount() >= 1; }));
  EXPECT_EQ(decode_->resultCount(), 0u);

  prefillA_->stop();

  ASSERT_TRUE(
      waitFor([&] { return decode_->resultCount() >= 1; }, /*timeout=*/3s))
      << "Decode should receive a failure result when prefill drops";

  auto results = decode_->results();
  ASSERT_EQ(results.size(), 1u);
  EXPECT_EQ(results[0].taskId, 55u);
  EXPECT_TRUE(results[0].error);
  EXPECT_EQ(results[0].generatedText, "prefill_down");
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

  auto results = decode.results();
  ASSERT_EQ(results.size(), 2u);
  for (const auto& result : results) {
    expectFakePrefillResultPayload(result);
  }

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

  gateway.registry().addCachedBlocks("prefill-A", {77});
  decode.sendRequest(/*task_id=*/303, /*hash=*/77);

  ASSERT_TRUE(waitFor([&] { return prefillA.receivedTaskCount() >= 1; }));

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
        router.rememberRegistration(msg.serverId, peerId);
        {
          std::lock_guard<std::mutex> lock(mutex);
          registered = true;
        }
        cv.notify_all();
      });

  router.registerHandler<tt::sockets::PrefillResultMessage>(
      tt::sockets::tags::PREFILL_RESULT,
      [&mutex, &cv, &resultReceived](
          const ZmqPrefillRouter::PeerIdentity&,
          const tt::sockets::PrefillResultMessage& msg) {
        EXPECT_EQ(msg.taskId, 7u);
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

      if (tt::sockets::wire::readMessageType(data) !=
          tt::sockets::tags::PREFILL_REQUEST) {
        continue;
      }

      auto request = tt::sockets::wire::deserializePayload<
          tt::sockets::PrefillRequestMessage>(data);
      EXPECT_EQ(request.taskId, 7u);
      {
        std::lock_guard<std::mutex> lock(mutex);
        requestReceived = true;
      }
      cv.notify_all();

      tt::sockets::PrefillResultMessage result(request.taskId);
      result.generatedText = "ok";
      client.sendRawData(tt::sockets::wire::serializeMessage(
          tt::sockets::tags::PREFILL_RESULT, result));
    }
  });

  tt::sockets::PrefillRegistrationMessage registration;
  registration.serverId = "prefill-A";
  registration.maxInFlight = 4;
  ASSERT_TRUE(client.sendRawData(tt::sockets::wire::serializeMessage(
      tt::sockets::tags::PREFILL_REGISTRATION, registration)));
  ASSERT_TRUE(waitFor([&] {
    std::lock_guard<std::mutex> lock(mutex);
    return registered;
  }));

  tt::sockets::PrefillRequestMessage request(7);
  ASSERT_TRUE(router.sendObject("prefill-A", tt::sockets::tags::PREFILL_REQUEST,
                                request));
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

TEST(ZmqSocketTransportTest, ServerIsConnectedOnlyAfterPeerIdentityIsKnown) {
  const uint16_t port = ephemeralPort();

  tt::sockets::ZmqSocketTransport server;
  ASSERT_TRUE(server.initializeAsServer(port));
  server.start();

  tt::sockets::ZmqSocketTransport client;
  ASSERT_TRUE(client.initializeAsClient("127.0.0.1", port));
  client.start();

  EXPECT_EQ(server.getStatus(), "server:waiting");
  EXPECT_FALSE(server.isConnected());

  tt::sockets::PrefillRegistrationMessage registration;
  registration.serverId = "prefill-A";
  registration.maxInFlight = 4;
  ASSERT_TRUE(client.sendRawData(tt::sockets::wire::serializeMessage(
      tt::sockets::tags::PREFILL_REGISTRATION, registration)));

  ASSERT_TRUE(waitFor([&] { return server.isConnected(); }));
  EXPECT_EQ(server.getStatus(), "server:connected");

  client.stop();
  server.stop();
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
        router.rememberRegistration(msg.serverId, peerId);
        std::lock_guard<std::mutex> lock(mutex);
        registered = true;
      });

  ASSERT_TRUE(router.start("127.0.0.1", port));

  tt::sockets::ZmqSocketTransport client;
  ASSERT_TRUE(client.initializeAsClient("127.0.0.1", port));
  client.start();

  tt::sockets::PrefillRegistrationMessage registration;
  registration.serverId = "prefill-A";
  registration.maxInFlight = 4;
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
        return !router.sendObject("prefill-A",
                                  tt::sockets::tags::PREFILL_REQUEST, request);
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
  const std::string healthResponse = httpGet(port, "/health");
  EXPECT_NE(healthResponse.find("HTTP/1.1 404 Not Found"), std::string::npos);

  server.stop();
}

TEST(GatewayHealthServerTest, ServesLivenessAndReadinessSeparately) {
  const uint16_t port = ephemeralPort();
  GatewayHealthServer server;
  server.setHealthProvider([] {
    GatewayHealthStatus status;
    status.livenessJson =
        R"({"status":"alive","transport":"tcp","registered_prefills":2,"healthy_prefills":2,"accepting_prefills":2,"decode_connected":false})"
        "\n";
    status.healthJson =
        R"({"status":"unhealthy","error":"decode not connected","transport":"tcp","registered_prefills":2,"healthy_prefills":2,"accepting_prefills":2,"decode_connected":false})"
        "\n";
    status.ready = false;
    status.error = "decode not connected";
    return status;
  });
  ASSERT_TRUE(server.start(port));

  ASSERT_TRUE(waitFor([&] {
    const std::string response = httpGet(port, "/tt-liveness");
    return response.find("HTTP/1.1 200 OK") != std::string::npos &&
           response.find("Content-Type: application/json") !=
               std::string::npos &&
           response.find(R"("status":"alive")") != std::string::npos &&
           response.find(R"("healthy_prefills":2)") != std::string::npos &&
           response.find(R"("decode_connected":false)") != std::string::npos;
  }));

  const std::string healthResponse = httpGet(port, "/health");
  EXPECT_NE(healthResponse.find("HTTP/1.1 503 Service Unavailable"),
            std::string::npos);
  EXPECT_NE(healthResponse.find(R"("status":"unhealthy")"), std::string::npos);
  EXPECT_NE(healthResponse.find(R"("error":"decode not connected")"),
            std::string::npos);

  server.stop();
}

}  // namespace
}  // namespace tt::gateway
