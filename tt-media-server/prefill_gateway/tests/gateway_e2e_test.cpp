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
  FakePrefill(std::string server_id, uint16_t port, uint32_t max_in_flight = 4)
      : server_id_(std::move(server_id)),
        port_(port),
        max_in_flight_(max_in_flight) {
    sm_.initializeAsServer(port_);

    sm_.setConnectionEstablishedCallback([this] {
      tt::sockets::PrefillRegistrationMessage msg;
      msg.server_id = server_id_;
      msg.max_in_flight = max_in_flight_;
      // Brief sleep — the gateway-side SERVER socket needs a moment to wire
      // its read loop before the registration message arrives.
      std::this_thread::sleep_for(50ms);
      sm_.sendObject(tt::sockets::tags::PREFILL_REGISTRATION, msg);
    });

    sm_.registerHandler<tt::sockets::PrefillRequestMessage>(
        "prefill_request",
        [this](const tt::sockets::PrefillRequestMessage& req) {
          received_task_ids_.fetch_add(1);
          {
            std::lock_guard<std::mutex> lock(mutex_);
            last_request_ = req;
          }
          if (!auto_reply_) return;
          tt::sockets::PrefillResultMessage res(req.task_id);
          res.error = false;
          res.finished = true;
          res.generated_text = "ok-from-" + server_id_;
          sm_.sendObject("prefill_result", res);
        });
  }

  ~FakePrefill() { sm_.stop(); }

  void start() { sm_.start(); }
  void stop() { sm_.stop(); }
  void setAutoReply(bool v) { auto_reply_ = v; }
  uint32_t receivedTaskCount() const { return received_task_ids_.load(); }
  const std::string& serverId() const { return server_id_; }

  std::optional<tt::sockets::PrefillRequestMessage> takeLastRequest() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!last_request_) return std::nullopt;
    auto out = *last_request_;
    last_request_.reset();
    return out;
  }

 private:
  std::string server_id_;
  uint16_t port_;
  uint32_t max_in_flight_;
  std::atomic<bool> auto_reply_{true};
  std::atomic<uint32_t> received_task_ids_{0};
  tt::sockets::SocketManager sm_;
  std::mutex mutex_;
  std::optional<tt::sockets::PrefillRequestMessage> last_request_;
};

// Mock decode: CLIENT to gateway, collects results/assignments.
class FakeDecode {
 public:
  FakeDecode(const std::string& gateway_host, uint16_t gateway_port) {
    sm_.initializeAsClient(gateway_host, gateway_port);
    sm_.setReconnectBackoff(50, 500);

    sm_.registerHandler<tt::sockets::PrefillResultMessage>(
        "prefill_result",
        [this](const tt::sockets::PrefillResultMessage& msg) {
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

  void sendRequest(uint32_t task_id, size_t registration_hash = 0) {
    tt::sockets::PrefillRequestMessage req(task_id);
    req.registration_hash = registration_hash;
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
  GatewayHarness(uint16_t decode_port,
                 const std::vector<std::pair<std::string, uint16_t>>& prefills)
      : decode_port_(decode_port) {
    decode_sm_.initializeAsServer(decode_port_);

    for (const auto& [host, port] : prefills) {
      auto sm = std::make_unique<tt::sockets::SocketManager>();
      sm->setReconnectBackoff(50, 500);
      sm->initializeAsClient(host, port);
      prefill_sms_.push_back(std::move(sm));
    }

    Dispatcher::Senders senders;
    senders.sendRequestToPrefill =
        [this](const std::string& server_id,
               const tt::sockets::PrefillRequestMessage& m) -> bool {
      auto* sm = registry_.getSocketManager(server_id);
      if (!sm) return false;
      return sm->sendObject("prefill_request", m);
    };
    senders.sendAssignmentToDecode =
        [this](const tt::sockets::PrefillAssignmentMessage& m) -> bool {
      return decode_sm_.sendObject(tt::sockets::tags::PREFILL_ASSIGNMENT, m);
    };
    senders.sendResultToDecode =
        [this](const tt::sockets::PrefillResultMessage& m) -> bool {
      return decode_sm_.sendObject("prefill_result", m);
    };

    dispatcher_ =
        std::make_unique<Dispatcher>(registry_, affinity_, std::move(senders));

    registry_.setOnPrefillDown([this](const std::string& id) {
      dispatcher_->onPrefillDown(id);
    });

    for (auto& sm_ptr : prefill_sms_) {
      auto* sm = sm_ptr.get();
      auto id_holder = std::make_shared<std::string>();

      sm->registerHandler<tt::sockets::PrefillRegistrationMessage>(
          tt::sockets::tags::PREFILL_REGISTRATION,
          [this, sm, id_holder](
              const tt::sockets::PrefillRegistrationMessage& msg) {
            *id_holder = msg.server_id;
            registry_.preRegister(msg.server_id, sm);
            registry_.markRegistered(msg.server_id, msg.max_in_flight);
          });

      sm->registerHandler<tt::sockets::PrefillResultMessage>(
          "prefill_result",
          [this, id_holder](const tt::sockets::PrefillResultMessage& msg) {
            dispatcher_->onPrefillResult(*id_holder, msg);
          });

      sm->setConnectionLostCallback([this, id_holder] {
        const std::string& sid = *id_holder;
        if (!sid.empty()) registry_.markDown(sid);
      });
    }

    decode_sm_.registerHandler<tt::sockets::PrefillRequestMessage>(
        "prefill_request",
        [this](const tt::sockets::PrefillRequestMessage& msg) {
          dispatcher_->onPrefillRequest(msg);
        });
  }

  ~GatewayHarness() {
    decode_sm_.stop();
    for (auto& sm : prefill_sms_) sm->stop();
  }

  void start() {
    for (auto& sm : prefill_sms_) sm->start();
    decode_sm_.start();
  }

  PrefillRegistry& registry() { return registry_; }
  AffinityCache& affinity() { return affinity_; }

 private:
  uint16_t decode_port_;
  PrefillRegistry registry_;
  AffinityCache affinity_;
  tt::sockets::SocketManager decode_sm_;
  std::vector<std::unique_ptr<tt::sockets::SocketManager>> prefill_sms_;
  std::unique_ptr<Dispatcher> dispatcher_;
};

class GatewayE2ETest : public ::testing::Test {
 protected:
  void SetUp() override {
    decode_port_ = ephemeralPort();
    prefill_a_port_ = ephemeralPort();
    prefill_b_port_ = ephemeralPort();

    prefill_a_ = std::make_unique<FakePrefill>("prefill-A", prefill_a_port_);
    prefill_b_ = std::make_unique<FakePrefill>("prefill-B", prefill_b_port_);
    prefill_a_->start();
    prefill_b_->start();

    gateway_ = std::make_unique<GatewayHarness>(
        decode_port_,
        std::vector<std::pair<std::string, uint16_t>>{
            {"127.0.0.1", prefill_a_port_},
            {"127.0.0.1", prefill_b_port_},
        });
    gateway_->start();

    decode_ = std::make_unique<FakeDecode>("127.0.0.1", decode_port_);
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
    prefill_a_.reset();
    prefill_b_.reset();
  }

  uint16_t decode_port_;
  uint16_t prefill_a_port_;
  uint16_t prefill_b_port_;
  std::unique_ptr<FakePrefill> prefill_a_;
  std::unique_ptr<FakePrefill> prefill_b_;
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
  uint32_t total = prefill_a_->receivedTaskCount() +
                   prefill_b_->receivedTaskCount();
  EXPECT_EQ(total, 1u);
}

TEST_F(GatewayE2ETest, StickyRoutingByRegistrationHash) {
  decode_->sendRequest(/*task_id=*/1, /*hash=*/42);
  ASSERT_TRUE(waitFor([&] { return decode_->resultCount() >= 1; }));
  auto first_assignments = decode_->assignments();
  ASSERT_EQ(first_assignments.size(), 1u);
  const std::string first_server = first_assignments[0].server_id;

  decode_->sendRequest(/*task_id=*/2, /*hash=*/42);
  ASSERT_TRUE(waitFor([&] { return decode_->resultCount() >= 2; }));
  auto all_assignments = decode_->assignments();
  ASSERT_EQ(all_assignments.size(), 2u);
  EXPECT_EQ(all_assignments[1].server_id, first_server)
      << "Sticky routing should reuse the previous prefill";
  EXPECT_EQ(all_assignments[1].task_id, 2u);
}

TEST_F(GatewayE2ETest, PrefillDownFailsInFlightTaskToDecode) {
  prefill_a_->setAutoReply(false);
  prefill_b_->setAutoReply(false);

  // Seed affinity so we know which prefill will take the request.
  gateway_->affinity().record(/*hash=*/77, "prefill-A");
  decode_->sendRequest(/*task_id=*/55, /*hash=*/77);

  ASSERT_TRUE(waitFor([&] { return prefill_a_->receivedTaskCount() >= 1; }));
  ASSERT_TRUE(waitFor([&] { return decode_->assignmentCount() >= 1; }));
  EXPECT_EQ(decode_->resultCount(), 0u);

  prefill_a_->stop();

  ASSERT_TRUE(waitFor(
      [&] { return decode_->resultCount() >= 1; }, /*timeout=*/3s))
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
