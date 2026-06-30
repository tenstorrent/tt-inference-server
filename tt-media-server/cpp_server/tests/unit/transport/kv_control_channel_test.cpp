// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_control_channel.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <deque>
#include <memory>
#include <span>
#include <string>
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

// Empty-but-connected reads throughout the timeout yield nullopt — but only
// after waiting, never aborting on the first transient empty read.
TEST(KvControlChannel, ReceiveTimesOutWhenNoData) {
  Queue a, b;
  auto t = std::make_shared<FakeTransport>(&a, &b);  // open, empty
  KvControlChannel ch(t, std::chrono::milliseconds(30),
                      std::chrono::milliseconds(1));
  EXPECT_FALSE(ch.receive().has_value());
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
