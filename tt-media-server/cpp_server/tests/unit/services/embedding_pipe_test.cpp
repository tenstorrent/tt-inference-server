// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Unit tests for the length-prefixed pipe framing between the embedding
// parent process and its forked workers (services/embedding_pipe.cpp).

#include "services/embedding_pipe.hpp"

#include <gtest/gtest.h>
#include <signal.h>
#include <unistd.h>

#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "config/defaults.hpp"

namespace pipe_detail = tt::services::embedding_detail;

namespace {

// RAII pipe(2) wrapper; tests can close either end early.
struct TestPipe {
  int readFd = -1;
  int writeFd = -1;

  TestPipe() {
    int raw[2] = {-1, -1};
    if (pipe(raw) == 0) {
      readFd = raw[0];
      writeFd = raw[1];
    }
  }

  ~TestPipe() {
    closeRead();
    closeWrite();
  }

  void closeRead() {
    if (readFd >= 0) {
      close(readFd);
      readFd = -1;
    }
  }

  void closeWrite() {
    if (writeFd >= 0) {
      close(writeFd);
      writeFd = -1;
    }
  }
};

// Deterministic non-trivial payload bytes.
std::vector<uint8_t> makePayload(size_t size) {
  std::vector<uint8_t> payload(size);
  for (size_t i = 0; i < size; ++i) {
    payload[i] = static_cast<uint8_t>(i * 31u + 7u);
  }
  return payload;
}

std::vector<uint8_t> toBytes(const std::string& s) {
  return {s.begin(), s.end()};
}

}  // namespace

// A written binary payload reads back byte-identical.
TEST(EmbeddingPipeTest, BinaryRoundTrip) {
  TestPipe p;
  const auto payload = makePayload(256);

  ASSERT_TRUE(pipe_detail::pipeWrite(p.writeFd, payload.data(), payload.size()));

  EXPECT_EQ(pipe_detail::pipeReadBinary(p.readFd), payload);
}

// Several messages on one pipe come back in order, without desync.
TEST(EmbeddingPipeTest, MultipleMessagesReadInOrder) {
  TestPipe p;
  const auto first = makePayload(8);
  const auto second = makePayload(1000);
  const std::string third = "third message";

  ASSERT_TRUE(pipe_detail::pipeWrite(p.writeFd, first.data(), first.size()));
  ASSERT_TRUE(pipe_detail::pipeWrite(p.writeFd, second.data(), second.size()));
  ASSERT_TRUE(pipe_detail::pipeWrite(p.writeFd, third.data(), third.size()));

  EXPECT_EQ(pipe_detail::pipeReadBinary(p.readFd), first);
  EXPECT_EQ(pipe_detail::pipeReadBinary(p.readFd), second);
  EXPECT_EQ(pipe_detail::pipeReadString(p.readFd), third);
}

// A written string payload reads back identical through pipeReadString.
TEST(EmbeddingPipeTest, StringRoundTrip) {
  TestPipe p;
  const std::string json = R"([{"model":"bge","input":"hello","task_id":1}])";

  ASSERT_TRUE(pipe_detail::pipeWrite(p.writeFd, json.data(), json.size()));

  EXPECT_EQ(pipe_detail::pipeReadString(p.readFd), json);
}

// EOF (writer closed with nothing written) yields an empty result — this is
// the worker's shutdown signal.
TEST(EmbeddingPipeTest, ReadBinaryReturnsEmptyOnEof) {
  TestPipe p;
  p.closeWrite();

  EXPECT_TRUE(pipe_detail::pipeReadBinary(p.readFd).empty());
}

// Same EOF behavior for the string variant.
TEST(EmbeddingPipeTest, ReadStringReturnsEmptyOnEof) {
  TestPipe p;
  p.closeWrite();

  EXPECT_TRUE(pipe_detail::pipeReadString(p.readFd).empty());
}

// A length prefix above EMBEDDING_MAX_PIPE_BYTES is rejected without
// allocating the claimed size.
TEST(EmbeddingPipeTest, ReadBinaryRejectsOversizedLengthPrefix) {
  TestPipe p;
  const uint32_t oversized =
      static_cast<uint32_t>(tt::config::defaults::EMBEDDING_MAX_PIPE_BYTES) + 1;
  ASSERT_EQ(write(p.writeFd, &oversized, sizeof(oversized)), 
            static_cast<ssize_t>(sizeof(oversized)));

  EXPECT_TRUE(pipe_detail::pipeReadBinary(p.readFd).empty());
}

// A header claiming more bytes than ever arrive fails cleanly once the
// writer closes, instead of hanging or returning a partial buffer.
TEST(EmbeddingPipeTest, TruncatedBinaryPayloadReturnsEmpty) {
  TestPipe p;
  const uint32_t claimed = 10;
  const char partial[4] = {'a', 'b', 'c', 'd'};
  ASSERT_EQ(write(p.writeFd, &claimed, sizeof(claimed)),
            static_cast<ssize_t>(sizeof(claimed)));
  ASSERT_EQ(write(p.writeFd, partial, sizeof(partial)),
            static_cast<ssize_t>(sizeof(partial)));
  p.closeWrite();

  EXPECT_TRUE(pipe_detail::pipeReadBinary(p.readFd).empty());
}

// Same truncation behavior for the string variant.
TEST(EmbeddingPipeTest, TruncatedStringPayloadReturnsEmpty) {
  TestPipe p;
  const uint32_t claimed = 10;
  const char partial[4] = {'a', 'b', 'c', 'd'};
  ASSERT_EQ(write(p.writeFd, &claimed, sizeof(claimed)),
            static_cast<ssize_t>(sizeof(claimed)));
  ASSERT_EQ(write(p.writeFd, partial, sizeof(partial)),
            static_cast<ssize_t>(sizeof(partial)));
  p.closeWrite();

  EXPECT_TRUE(pipe_detail::pipeReadString(p.readFd).empty());
}

// A payload much larger than the kernel pipe buffer (64 KB on Linux) forces
// partial reads; the reassembly loop must still produce identical bytes.
TEST(EmbeddingPipeTest, LargePayloadIsReassembledFromPartialReads) {
  TestPipe p;
  const auto payload = makePayload(1 << 20);  // 1 MB

  std::thread writer([&] {
    pipe_detail::pipeWrite(p.writeFd, payload.data(), payload.size());
  });
  const auto result = pipe_detail::pipeReadBinary(p.readFd);
  writer.join();

  EXPECT_EQ(result, payload);
}

// The payload arriving in small delayed chunks (header first, body dribbled)
// exercises short reads deterministically.
TEST(EmbeddingPipeTest, ChunkedWriteIsReassembled) {
  TestPipe p;
  const std::string message = "chunked-message-payload";

  std::thread writer([&] {
    const uint32_t len = static_cast<uint32_t>(message.size());
    write(p.writeFd, &len, sizeof(len));
    for (char c : message) {
      write(p.writeFd, &c, 1);
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
  });
  const auto result = pipe_detail::pipeReadString(p.readFd);
  writer.join();

  EXPECT_EQ(result, message);
}

// Documents a known wart: a zero-length message is a valid write but reads
// back as an empty buffer, indistinguishable from failure/EOF for callers.
TEST(EmbeddingPipeTest, ZeroLengthMessageReadsBackEmpty) {
  TestPipe p;

  ASSERT_TRUE(pipe_detail::pipeWrite(p.writeFd, "", 0));

  EXPECT_TRUE(pipe_detail::pipeReadBinary(p.readFd).empty());
}

// Writing after the reader closed its end fails instead of crashing
// (SIGPIPE ignored, as a dead worker must not take the parent down).
TEST(EmbeddingPipeTest, WriteFailsWhenReadEndClosed) {
  signal(SIGPIPE, SIG_IGN);
  TestPipe p;
  p.closeRead();
  const auto payload = makePayload(16);

  EXPECT_FALSE(pipe_detail::pipeWrite(p.writeFd, payload.data(), payload.size()));
}

// Only the exact READY payload is accepted as the warmup handshake.
TEST(EmbeddingPipeTest, ReadySentinelMatchesExactPayloadOnly) {
  EXPECT_TRUE(pipe_detail::isReadySentinel(toBytes("READY")));

  EXPECT_FALSE(pipe_detail::isReadySentinel(toBytes("READYX")));
  EXPECT_FALSE(pipe_detail::isReadySentinel(toBytes("READ")));
  EXPECT_FALSE(pipe_detail::isReadySentinel(toBytes("REKDY")));
  EXPECT_FALSE(pipe_detail::isReadySentinel({}));
}

// The sentinel round-trips through the pipe exactly as the worker sends it
// after warmup (embedding_worker_main.cpp).
TEST(EmbeddingPipeTest, ReadySentinelSurvivesPipeRoundTrip) {
  TestPipe p;
  ASSERT_TRUE(pipe_detail::pipeWrite(p.writeFd, pipe_detail::WORKER_READY_SENTINEL,
                                     sizeof(pipe_detail::WORKER_READY_SENTINEL) - 1));

  EXPECT_TRUE(pipe_detail::isReadySentinel(pipe_detail::pipeReadBinary(p.readFd)));
}
