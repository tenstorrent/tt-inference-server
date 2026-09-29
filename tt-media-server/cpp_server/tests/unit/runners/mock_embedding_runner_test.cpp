// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Unit tests for MockEmbeddingRunner, the no-Python no-device stand-in that
// CI and hardware-less development run against.

#include "runtime/runners/mock_embedding_runner.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"

using tt::config::EmbeddingConfig;
using tt::domain::EmbeddingRequest;
using tt::domain::EmbeddingResponse;
using tt::runners::MockEmbeddingRunner;

namespace {

constexpr char kModel[] = "BAAI/bge-large-en-v1.5";
constexpr size_t kDim = 1024;  // the mock's fixed output dimension

EmbeddingConfig makeConfig(size_t maxBatchSize = 8) {
  EmbeddingConfig cfg;
  cfg.hf_model_id = kModel;
  cfg.max_batch_size = maxBatchSize;
  return cfg;
}

EmbeddingRequest makeRequest(uint32_t taskId, const std::string& input) {
  EmbeddingRequest req(taskId);
  req.model = kModel;
  req.input = input;
  return req;
}

bool bitIdentical(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace

// Warmup has nothing to load and must report success.
TEST(MockEmbeddingRunnerTest, WarmupSucceeds) {
  MockEmbeddingRunner runner(makeConfig());
  EXPECT_TRUE(runner.warmup());
}

// The same input yields a bit-identical vector across calls and across
// runner instances — the mock's core determinism promise.
TEST(MockEmbeddingRunnerTest, SameInputIsDeterministic) {
  MockEmbeddingRunner first(makeConfig());
  MockEmbeddingRunner second(makeConfig());
  const std::vector<EmbeddingRequest> batch{makeRequest(1, "hello world")};

  const auto a = first.run(batch);
  const auto b = first.run(batch);
  const auto c = second.run(batch);

  ASSERT_EQ(a.size(), 1u);
  EXPECT_TRUE(bitIdentical(a[0].embedding, b[0].embedding));
  EXPECT_TRUE(bitIdentical(a[0].embedding, c[0].embedding));
}

// Golden values: the hash/PRNG constants are chosen for cross-machine
// reproducibility; changing them silently invalidates vectors captured
// elsewhere, so a few values are pinned here.
TEST(MockEmbeddingRunnerTest, KnownInputMatchesPinnedValues) {
  MockEmbeddingRunner runner(makeConfig());

  const auto responses = runner.run({makeRequest(1, "hello world")});

  ASSERT_EQ(responses.size(), 1u);
  const auto& v = responses[0].embedding;
  ASSERT_EQ(v.size(), kDim);
  EXPECT_FLOAT_EQ(v[0], -0.0166425277f);
  EXPECT_FLOAT_EQ(v[1], -0.0156397987f);
  EXPECT_FLOAT_EQ(v[511], -0.0195555873f);
  EXPECT_FLOAT_EQ(v[1023], -0.0456175357f);
  EXPECT_EQ(responses[0].total_tokens, 4);  // "hello world": 2 words + 2
}

// Distinct inputs must produce clearly different vectors.
TEST(MockEmbeddingRunnerTest, DistinctInputsYieldDistinctVectors) {
  MockEmbeddingRunner runner(makeConfig());

  const auto responses = runner.run(
      {makeRequest(1, "first text"), makeRequest(2, "second text")});

  ASSERT_EQ(responses.size(), 2u);
  EXPECT_FALSE(bitIdentical(responses[0].embedding, responses[1].embedding));
}

// Output vectors are unit-length, so cosine comparisons behave sensibly.
TEST(MockEmbeddingRunnerTest, VectorsAreUnitNorm) {
  MockEmbeddingRunner runner(makeConfig());

  const auto responses = runner.run({makeRequest(1, "normalize me")});

  ASSERT_EQ(responses.size(), 1u);
  double sumSquares = 0.0;
  for (float x : responses[0].embedding) sumSquares += double(x) * x;
  EXPECT_NEAR(sumSquares, 1.0, 1e-4);
}

// Every response echoes its request's task_id and model, in order.
TEST(MockEmbeddingRunnerTest, EchoesTaskIdAndModelPositionally) {
  MockEmbeddingRunner runner(makeConfig());

  const auto responses =
      runner.run({makeRequest(10, "a"), makeRequest(11, "b")});

  ASSERT_EQ(responses.size(), 2u);
  EXPECT_EQ(responses[0].task_id, 10u);
  EXPECT_EQ(responses[1].task_id, 11u);
  EXPECT_EQ(responses[0].model, kModel);
  EXPECT_TRUE(responses[0].error.empty());
}

// A batch above max_batch_size fails every request with the same error,
// mirroring the real runner's Python-side assertion.
TEST(MockEmbeddingRunnerTest, OversizedBatchFailsEveryRequest) {
  MockEmbeddingRunner runner(makeConfig(2));

  const auto responses = runner.run(
      {makeRequest(1, "a"), makeRequest(2, "b"), makeRequest(3, "c")});

  ASSERT_EQ(responses.size(), 3u);
  for (const auto& r : responses) {
    EXPECT_EQ(r.error, "Batch size 3 exceeds max 2");
    EXPECT_TRUE(r.embedding.empty());
  }
}

// An unknown model name fails with the exact message the real runner uses.
// Note one deliberate divergence: the mock fails only the mismatched request,
// while the real runner fails the whole batch.
TEST(MockEmbeddingRunnerTest, WrongModelFailsOnlyThatRequest) {
  MockEmbeddingRunner runner(makeConfig());
  auto bad = makeRequest(2, "b");
  bad.model = "some/other-model";

  const auto responses = runner.run({makeRequest(1, "a"), bad});

  ASSERT_EQ(responses.size(), 2u);
  EXPECT_TRUE(responses[0].error.empty());
  EXPECT_EQ(responses[1].error,
            std::string("Only ") + kModel + " embeddings are supported");
  EXPECT_TRUE(responses[1].embedding.empty());
}

// Token count approximates a BERT-style tokenizer: words + 2 specials,
// clamped at the mock's 384 sequence limit.
TEST(MockEmbeddingRunnerTest, TokenCountApproximation) {
  MockEmbeddingRunner runner(makeConfig());

  auto tokensFor = [&](const std::string& input) {
    return runner.run({makeRequest(1, input)})[0].total_tokens;
  };

  EXPECT_EQ(tokensFor("one two three"), 5);
  EXPECT_EQ(tokensFor(""), 2);
  EXPECT_EQ(tokensFor("  \t\n  "), 2);
  EXPECT_EQ(tokensFor("spaced\t\tout\n\nwords"), 5);

  std::string longInput;
  for (int i = 0; i < 500; ++i) longInput += "word ";
  EXPECT_EQ(tokensFor(longInput), 384);
}

// An empty batch yields an empty response list.
TEST(MockEmbeddingRunnerTest, EmptyBatchYieldsEmptyResponses) {
  MockEmbeddingRunner runner(makeConfig());
  EXPECT_TRUE(runner.run({}).empty());
}
