// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_control_channel.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "sockets/i_socket_transport.hpp"
#include "transport/kv_control_message.hpp"

namespace tt::transport {
namespace {

using Queue = std::deque<std::vector<uint8_t>>;

// In-memory loopback transport: sends go to the peer's inbox, receives pop from
// our own inbox. Two of them sharing crossed queues form a connected pair.
//
// Like the real TcpSocketTransport, tryReceiveMessage() reports the genuine
// tri-state directly — DATA when the inbox has a message, NO_DATA while still
// connected but nothing is ready, CLOSED once dropped — so the channel relies
// on that distinction rather than on isConnected(). `stall` yields NO_DATA for
// a few reads to simulate a slow peer; `closed` drops the connection.
class FakeTransport : public sockets::ISocketTransport {
 public:
  FakeTransport(Queue* inbox, Queue* peerInbox)
      : inbox(inbox), peer_inbox(peerInbox) {}

  bool closed = false;
  int stall = 0;  ///< NO_DATA reads before serving the inbox.

  bool initializeAsServer(uint16_t) override { return true; }
  bool initializeAsClient(const std::string&, uint16_t) override {
    return true;
  }
  void start() override {}
  void stop() override {}
  bool isConnected() const override { return !closed; }
  std::string getStatus() const override { return "fake"; }

  bool sendRawData(std::span<const uint8_t> data) override {
    peer_inbox->emplace_back(data.begin(), data.end());
    return true;
  }
  std::vector<uint8_t> receiveRawData() override {
    return std::move(tryReceiveMessage().data);
  }
  sockets::ReceiveResult tryReceiveMessage() override {
    if (stall > 0) {
      --stall;
      return {sockets::ReceiveStatus::NO_DATA, {}};  // connected, not ready yet
    }
    if (inbox->empty()) {
      return {closed ? sockets::ReceiveStatus::CLOSED
                     : sockets::ReceiveStatus::NO_DATA,
              {}};
    }
    std::vector<uint8_t> front = std::move(inbox->front());
    inbox->pop_front();
    return {sockets::ReceiveStatus::DATA, std::move(front)};
  }
  void setConnectionLostCallback(std::function<void()>) override {}
  void setConnectionEstablishedCallback(std::function<void()>) override {}

 private:
  Queue* inbox;
  Queue* peer_inbox;
};

KvControlMessage beginMsg() {
  KvControlMessage m;
  m.type = KvControlType::BEGIN_MIGRATION;
  m.uuid = 0xDEADBEEFCAFEULL;
  m.slot = 5;
  m.layer_begin = 0;
  m.layer_end = 61;
  m.position_begin = 0;
  m.position_end = 4096;
  return m;
}

// Every message kind survives a serialize/deserialize round-trip.
TEST(KvControlMessage, RoundTripsAllKinds) {
  std::vector<KvControlMessage> kinds;
  kinds.push_back(beginMsg());

  KvControlMessage table;
  table.type = KvControlType::TABLE_EXCHANGE;
  table.role = 1;
  table.table_blob = {1, 2, 3, 4, 5, 0, 255, 128};
  kinds.push_back(table);

  KvControlMessage ready;
  ready.type = KvControlType::MIRROR_READY;
  ready.uuid = 7;
  ready.segment_name = "decode-A:127.0.0.1:18632";
  kinds.push_back(ready);

  KvControlMessage done;
  done.type = KvControlType::DONE_MARKER;
  done.uuid = 7;
  kinds.push_back(done);

  KvControlMessage ack;
  ack.type = KvControlType::ACK;
  ack.uuid = 7;
  kinds.push_back(ack);

  for (const auto& m : kinds) {
    const auto bytes = m.serialize();
    const auto parsed = KvControlMessage::deserialize(bytes);
    ASSERT_TRUE(parsed.has_value());
    EXPECT_TRUE(*parsed == m);
  }
}

// Malformed / truncated buffers are rejected, not misparsed.
TEST(KvControlMessage, RejectsMalformed) {
  EXPECT_FALSE(KvControlMessage::deserialize({}).has_value());
  EXPECT_FALSE(KvControlMessage::deserialize(std::vector<uint8_t>{99})
                   .has_value());  // bad type

  auto good = beginMsg().serialize();
  good.resize(good.size() - 1);  // truncate the trailing length-prefixed blob
  EXPECT_FALSE(KvControlMessage::deserialize(good).has_value());
}

// A message sent on one end of a connected channel pair arrives intact.
TEST(KvControlChannel, LoopbackDeliversMessage) {
  Queue aToB, bToA;
  auto sender = std::make_shared<FakeTransport>(&bToA, &aToB);
  auto receiver = std::make_shared<FakeTransport>(&aToB, &bToA);
  KvControlChannel senderCh(sender);
  KvControlChannel receiverCh(receiver);

  EXPECT_TRUE(senderCh.isConnected());
  ASSERT_TRUE(senderCh.send(beginMsg()));

  const auto got = receiverCh.receive();
  ASSERT_TRUE(got.has_value());
  EXPECT_TRUE(*got == beginMsg());

  // A reply flows back the other way.
  KvControlMessage ready;
  ready.type = KvControlType::MIRROR_READY;
  ready.uuid = beginMsg().uuid;
  ready.segment_name = "seg-1";
  ASSERT_TRUE(receiverCh.send(ready));
  const auto reply = senderCh.receive();
  ASSERT_TRUE(reply.has_value());
  EXPECT_EQ(reply->segment_name, "seg-1");
}

// B2: a Transaction held across send+receive must block try_lock from another
// thread — migrate and TABLE_EXCHANGE must not interleave message boundaries.
TEST(KvControlChannel, TransactionTryLockFailsWhileHeld) {
  Queue aToB, bToA;
  auto sender = std::make_shared<FakeTransport>(&bToA, &aToB);
  KvControlChannel ch(sender);

  KvControlChannel::Transaction held(ch);
  ASSERT_TRUE(held.ownsLock());

  std::atomic<bool> tryStarted{false};
  std::atomic<bool> tryOwned{true};
  std::thread other([&] {
    tryStarted.store(true);
    KvControlChannel::Transaction probe(ch, std::try_to_lock);
    tryOwned.store(probe.ownsLock());
  });
  while (!tryStarted.load()) {
    std::this_thread::yield();
  }
  // Give the other thread a moment to attempt try_lock while we hold.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  other.join();
  EXPECT_FALSE(tryOwned.load());
}

TEST(KvControlChannel, TransactionTryLockSucceedsWhenFree) {
  Queue aToB, bToA;
  auto sender = std::make_shared<FakeTransport>(&bToA, &aToB);
  KvControlChannel ch(sender);

  KvControlChannel::Transaction probe(ch, std::try_to_lock);
  EXPECT_TRUE(probe.ownsLock());
}

// A closed transport (tryReceiveMessage reports CLOSED) yields nullopt
// immediately, not a bogus message and not a wait.
TEST(KvControlChannel, ReceiveClosedIsNullopt) {
  Queue a, b;
  auto t = std::make_shared<FakeTransport>(&a, &b);
  t->closed = true;
  KvControlChannel ch(t, std::chrono::milliseconds(50),
                      std::chrono::milliseconds(1));
  EXPECT_FALSE(ch.receive().has_value());
}

// Empty-but-connected reads throughout the timeout yield TimedOut — never
// Closed — so decode's idle serve loop keeps the session (no reconnect /
// TABLE_EXCHANGE).
TEST(KvControlChannel, ReceiveTimesOutWhenNoData) {
  Queue a, b;
  auto t = std::make_shared<FakeTransport>(&a, &b);  // open, empty
  KvControlChannel ch(t, std::chrono::milliseconds(30),
                      std::chrono::milliseconds(1));
  KvControlMessage msg;
  EXPECT_EQ(ch.receiveMessage(msg), KvControlChannel::ReceiveOutcome::TimedOut);
  EXPECT_TRUE(ch.isConnected());
}

// Models the pre-fix TcpSocketTransport bug: IoBudget expiry on an idle
// probe was reported as CLOSED (markDisconnected). The channel must check its
// deadline before probing so a poll sleep that crosses the budget never
// turns TimedOut into Closed.
class BudgetDisconnectFake : public sockets::ISocketTransport {
 public:
  BudgetDisconnectFake() = default;

  bool initializeAsServer(uint16_t) override { return true; }
  bool initializeAsClient(const std::string&, uint16_t) override {
    return true;
  }
  void start() override {}
  void stop() override {}
  bool isConnected() const override { return !closed; }
  std::string getStatus() const override { return "budget-fake"; }

  void beginIoBudget(std::chrono::milliseconds budget) override {
    if (budget.count() <= 0) {
      clearIoBudget();
      return;
    }
    deadline = std::chrono::steady_clock::now() + budget;
  }
  void clearIoBudget() override { deadline.reset(); }

  bool sendRawData(std::span<const uint8_t>) override { return true; }
  std::vector<uint8_t> receiveRawData() override {
    return std::move(tryReceiveMessage().data);
  }
  sockets::ReceiveResult tryReceiveMessage() override {
    if (deadline && std::chrono::steady_clock::now() >= *deadline) {
      closed = true;
      return {sockets::ReceiveStatus::CLOSED, {}};
    }
    return {sockets::ReceiveStatus::NO_DATA, {}};
  }
  void setConnectionLostCallback(std::function<void()>) override {}
  void setConnectionEstablishedCallback(std::function<void()>) override {}

 private:
  bool closed = false;
  std::optional<std::chrono::steady_clock::time_point> deadline;
};

TEST(KvControlChannel, IdleTimeoutDoesNotDisconnect) {
  auto t = std::make_shared<BudgetDisconnectFake>();
  // poll_interval larger than remaining budget after first NO_DATA so the
  // sleep overshoots the IoBudget — the bug path without a pre-probe check.
  KvControlChannel ch(t, std::chrono::milliseconds(25),
                      std::chrono::milliseconds(40));
  KvControlMessage msg;
  EXPECT_EQ(ch.receiveMessage(msg), KvControlChannel::ReceiveOutcome::TimedOut);
  EXPECT_TRUE(ch.isConnected())
      << "idle TimedOut must not tear the transport down";
}

// The regression for the reported race: the peer replies after several empty
// (but connected) reads — it was still preparing. receive() must wait through
// them and deliver, not misread the gap as a close and abort.
TEST(KvControlChannel, ReceiveWaitsThroughNoDataThenDelivers) {
  Queue inbox, peer;
  KvControlMessage ready;
  ready.type = KvControlType::MIRROR_READY;
  ready.uuid = beginMsg().uuid;
  ready.segment_name = "seg-after-delay";
  inbox.emplace_back(ready.serialize());

  auto t = std::make_shared<FakeTransport>(&inbox, &peer);
  t->stall = 5;  // five empty (connected) reads before the message is served
  KvControlChannel ch(t, std::chrono::seconds(2), std::chrono::milliseconds(1));

  const auto got = ch.receive();
  ASSERT_TRUE(got.has_value());
  EXPECT_EQ(got->type, KvControlType::MIRROR_READY);
  EXPECT_EQ(got->segment_name, "seg-after-delay");
}

}  // namespace
}  // namespace tt::transport
