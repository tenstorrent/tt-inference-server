// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Unit tests for the binary pipe codec used between EmbeddingService and its
// forked workers (services/embedding_codec.hpp).
//
// The codec's contract:
//   - encodeResponses pairs batch[i] with responses[i] POSITIONALLY and stamps
//     each wire entry with batch[i].task_id. A bug here hands one caller
//     another caller's embedding, silently.
//   - decodeResponses returns a map keyed by task_id.
//   - A response list shorter than the batch yields "No response from runner"
//     error entries for the missing tail.
//   - Floats must round-trip BIT-EXACTLY; golden-vector validation relies on
//     byte-identical embeddings surviving the pipe.
//
// Known limitation (documented deliberately, not tested): detail::Reader does
// no bounds checking, so a buffer truncated in the MIDDLE of an entry is
// undefined behavior. The only safe truncation is between entries, which the
// atEnd() check in the decode loop handles; that case is tested below.
// Hardening the Reader is deferred to the Phase 4 rewrite so that this test
// file stays a pure observer of current behavior.

#include "services/embedding_codec.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"

namespace codec = tt::services::embedding_codec;
using tt::domain::EmbeddingRequest;
using tt::domain::EmbeddingResponse;

namespace {

EmbeddingRequest makeRequest(uint32_t taskId, const std::string& input) {
  EmbeddingRequest req(taskId);
  req.model = "BAAI/bge-large-en-v1.5";
  req.input = input;
  return req;
}

EmbeddingResponse makeSuccess(uint32_t taskId, std::vector<float> embedding,
                              int totalTokens) {
  EmbeddingResponse resp(taskId);
  resp.embedding = std::move(embedding);
  resp.total_tokens = totalTokens;
  resp.model = "BAAI/bge-large-en-v1.5";
  return resp;
}

EmbeddingResponse makeError(uint32_t taskId, const std::string& error) {
  EmbeddingResponse resp(taskId);
  resp.error = error;
  return resp;
}

// Deterministic pseudo-embedding: distinct per seed, includes negatives and
// values that exercise float bit patterns.
std::vector<float> makeVector(size_t dim, uint32_t seed) {
  std::vector<float> v(dim);
  for (size_t i = 0; i < dim; ++i) {
    v[i] = (static_cast<float>((seed * 31u + i) % 997u) - 498.5f) / 498.5f;
  }
  return v;
}

bool bitIdentical(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  return a.empty() ||
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

TEST(EmbeddingCodecTest, RoundTripSingleSuccess) {
  const auto vec = makeVector(1024, 7);
  std::vector<EmbeddingRequest> batch{makeRequest(42, "hello world")};
  std::vector<EmbeddingResponse> responses{makeSuccess(42, vec, 12)};

  auto decoded =
      codec::decodeResponses(codec::encodeResponses(batch, responses));

  ASSERT_EQ(decoded.size(), 1u);
  ASSERT_TRUE(decoded.count(42));
  const auto& r = decoded.at(42);
  EXPECT_TRUE(r.error.empty());
  EXPECT_TRUE(bitIdentical(r.embedding, vec));
  EXPECT_EQ(r.total_tokens, 12);
  EXPECT_EQ(r.model, "BAAI/bge-large-en-v1.5");
}

TEST(EmbeddingCodecTest, BatchPairsEachTaskIdWithItsOwnVector) {
  constexpr size_t kBatch = 8;
  std::vector<EmbeddingRequest> batch;
  std::vector<EmbeddingResponse> responses;
  for (size_t i = 0; i < kBatch; ++i) {
    const auto taskId = static_cast<uint32_t>(100 + i);
    batch.push_back(makeRequest(taskId, "prompt " + std::to_string(i)));
    responses.push_back(makeSuccess(taskId, makeVector(1024, taskId),
                                    static_cast<int>(10 + i)));
  }

  auto decoded =
      codec::decodeResponses(codec::encodeResponses(batch, responses));

  ASSERT_EQ(decoded.size(), kBatch);
  for (size_t i = 0; i < kBatch; ++i) {
    const auto taskId = static_cast<uint32_t>(100 + i);
    ASSERT_TRUE(decoded.count(taskId)) << "missing task_id " << taskId;
    const auto& r = decoded.at(taskId);
    EXPECT_TRUE(bitIdentical(r.embedding, makeVector(1024, taskId)))
        << "task_id " << taskId << " got a different task's vector";
    EXPECT_EQ(r.total_tokens, static_cast<int>(10 + i));
  }
}

TEST(EmbeddingCodecTest, ErrorEntryRoundTrip) {
  std::vector<EmbeddingRequest> batch{makeRequest(7, "boom")};
  std::vector<EmbeddingResponse> responses{makeError(7, "runner exploded")};

  auto decoded =
      codec::decodeResponses(codec::encodeResponses(batch, responses));

  ASSERT_EQ(decoded.size(), 1u);
  const auto& r = decoded.at(7);
  EXPECT_EQ(r.error, "runner exploded");
  EXPECT_TRUE(r.embedding.empty());
}

TEST(EmbeddingCodecTest, ShortResponseListFillsMissingTailWithErrors) {
  std::vector<EmbeddingRequest> batch{makeRequest(1, "a"), makeRequest(2, "b"),
                                      makeRequest(3, "c")};
  // Only the first request got a result from the runner.
  std::vector<EmbeddingResponse> responses{
      makeSuccess(1, makeVector(16, 1), 3)};

  auto decoded =
      codec::decodeResponses(codec::encodeResponses(batch, responses));

  ASSERT_EQ(decoded.size(), 3u);
  EXPECT_TRUE(decoded.at(1).error.empty());
  EXPECT_EQ(decoded.at(2).error, "No response from runner");
  EXPECT_EQ(decoded.at(3).error, "No response from runner");
}

TEST(EmbeddingCodecTest, MixedSuccessAndErrorPreservesBoth) {
  std::vector<EmbeddingRequest> batch{
      makeRequest(10, "ok"), makeRequest(11, "fail"), makeRequest(12, "ok")};
  std::vector<EmbeddingResponse> responses{
      makeSuccess(10, makeVector(64, 10), 5), makeError(11, "tokenizer error"),
      makeSuccess(12, makeVector(64, 12), 9)};

  auto decoded =
      codec::decodeResponses(codec::encodeResponses(batch, responses));

  ASSERT_EQ(decoded.size(), 3u);
  EXPECT_TRUE(decoded.at(10).error.empty());
  EXPECT_EQ(decoded.at(11).error, "tokenizer error");
  EXPECT_TRUE(decoded.at(12).error.empty());
  EXPECT_TRUE(bitIdentical(decoded.at(12).embedding, makeVector(64, 12)));
}

TEST(EmbeddingCodecTest, EmptyBatchProducesEmptyMap) {
  auto decoded = codec::decodeResponses(codec::encodeResponses({}, {}));
  EXPECT_TRUE(decoded.empty());
}

TEST(EmbeddingCodecTest, EmptyEmbeddingVectorRoundTrips) {
  std::vector<EmbeddingRequest> batch{makeRequest(5, "x")};
  std::vector<EmbeddingResponse> responses{makeSuccess(5, {}, 1)};

  auto decoded =
      codec::decodeResponses(codec::encodeResponses(batch, responses));

  ASSERT_EQ(decoded.size(), 1u);
  EXPECT_TRUE(decoded.at(5).error.empty());
  EXPECT_TRUE(decoded.at(5).embedding.empty());
}

TEST(EmbeddingCodecTest, LargeBatchWithRealisticDimsRoundTrips) {
  // 32 entries (the MAX_IN_FLIGHT_COUNT default) at BGE's 1024 dims.
  constexpr size_t kBatch = 32;
  std::vector<EmbeddingRequest> batch;
  std::vector<EmbeddingResponse> responses;
  for (size_t i = 0; i < kBatch; ++i) {
    const auto taskId = static_cast<uint32_t>(1000 + i);
    batch.push_back(makeRequest(taskId, std::string(384, 'w')));
    responses.push_back(makeSuccess(taskId, makeVector(1024, taskId), 384));
  }

  auto decoded =
      codec::decodeResponses(codec::encodeResponses(batch, responses));

  ASSERT_EQ(decoded.size(), kBatch);
  for (const auto& [taskId, r] : decoded) {
    EXPECT_TRUE(bitIdentical(r.embedding, makeVector(1024, taskId)));
  }
}

// The one SAFE truncation: the buffer ends cleanly BETWEEN entries while the
// leading count claims more. The atEnd() check must stop the loop after the
// complete entries. (Truncation mid-entry is UB in the current Reader — see
// the header comment — and is intentionally not exercised.)
TEST(EmbeddingCodecTest, CountLargerThanBufferStopsAtCleanEntryBoundary) {
  std::vector<EmbeddingRequest> batch{makeRequest(1, "a"), makeRequest(2, "b")};
  std::vector<EmbeddingResponse> responses{makeSuccess(1, makeVector(8, 1), 3),
                                           makeSuccess(2, makeVector(8, 2), 4)};

  auto buffer = codec::encodeResponses(batch, responses);
  // Inflate the declared count from 2 to 5 without adding entry bytes.
  const uint32_t inflated = 5;
  std::memcpy(buffer.data(), &inflated, sizeof(inflated));

  auto decoded = codec::decodeResponses(buffer);

  EXPECT_EQ(decoded.size(), 2u);
  EXPECT_TRUE(decoded.count(1));
  EXPECT_TRUE(decoded.count(2));
}
