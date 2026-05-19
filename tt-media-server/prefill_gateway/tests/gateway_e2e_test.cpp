// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// End-to-end integration test for PrefillGateway over real loopback sockets.
// Validates registration handshake, routing (round-robin + sticky-by-hash),
// result/assignment delivery, and prefill-down failover.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gateway/affinity_cache.hpp"
#include "gateway/dispatcher.hpp"
#include "gateway/prefill_registry.hpp"
#include "sockets/socket_manager.hpp"
#include "sockets/socket_messages.hpp"

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

// Mock prefill: SERVER mode, registers on connect, echoes a result per request.
class FakePrefill {
 public:
  FakePrefill(std::string serverId, uint16_t port, uint32_t maxInFlight = 4)
      : serverId_(std::move(serverId)), port_(port), maxInFlight_(maxInFlight) {
    sm_.initializeAsServer(port_);

    sm_.setConnectionEstablishedCallback([this] {
      tt::sockets::PrefillRegistrationMessage msg;
      msg.server_id = serverId_;
      msg.max_in_flight = maxInFlight_;
      // Brief sleep — the gateway-side SERVER socket needs a moment to wire
      // its read loop before the registration message arrives.
      std::this_thread::sleep_for(50ms);
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
  }

  ~FakePrefill() { sm_.stop(); }

  void start() { sm_.start(); }
  void stop() { sm_.stop(); }
  void setAutoReply(bool v) { autoReply_ = v; }
  uint32_t receivedTaskCount() const { return receivedTaskIds_.load(); }
  const std::string& serverId() const { return serverId_; }

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
};

// Mock decode: CLIENT to gateway, collects results/assignments.
class FakeDecode {
 public:
  FakeDecode(const std::string& gatewayHost, uint16_t gatewayPort) {
    sm_.initializeAsClient(gatewayHost, gatewayPort);
    sm_.setReconnectBackoff(50, 500);

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
    req.registration_hash = registrationHash;
    sm_.sendObject("prefill_request", req);
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
      sm->setReconnectBackoff(50, 500);
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
      auto idHolder = std::make_shared<std::string>();

      sm->registerHandler<tt::sockets::PrefillRegistrationMessage>(
          tt::sockets::tags::PREFILL_REGISTRATION,
          [this, sm,
           idHolder](const tt::sockets::PrefillRegistrationMessage& msg) {
            *idHolder = msg.server_id;
            registry_.preRegister(msg.server_id, sm);
            registry_.markRegistered(msg.server_id, msg.max_in_flight);
          });

      sm->registerHandler<tt::sockets::PrefillResultMessage>(
          "prefill_result",
          [this, idHolder](const tt::sockets::PrefillResultMessage& msg) {
            dispatcher_->onPrefillResult(*idHolder, msg);
          });

      sm->setConnectionLostCallback([this, idHolder] {
        const std::string& sid = *idHolder;
        if (!sid.empty()) registry_.markDown(sid);
      });
    }

    decodeSm_.registerHandler<tt::sockets::PrefillRequestMessage>(
        "prefill_request",
        [this](const tt::sockets::PrefillRequestMessage& msg) {
          dispatcher_->onPrefillRequest(msg);
        });
  }

  ~GatewayHarness() {
    decodeSm_.stop();
    for (auto& sm : prefillSms_) sm->stop();
  }

  void start() {
    for (auto& sm : prefillSms_) sm->start();
    decodeSm_.start();
  }

  PrefillRegistry& registry() { return registry_; }
  AffinityCache& affinity() { return affinity_; }

 private:
  uint16_t decodePort_;
  PrefillRegistry registry_;
  AffinityCache affinity_;
  tt::sockets::SocketManager decodeSm_;
  std::vector<std::unique_ptr<tt::sockets::SocketManager>> prefillSms_;
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

}  // namespace
}  // namespace tt::gateway
