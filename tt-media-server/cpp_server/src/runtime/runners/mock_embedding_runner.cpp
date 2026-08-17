// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/mock_embedding_runner.hpp"

#include <cmath>
#include <cstdint>
#include <string>

#include "utils/logger.hpp"

namespace tt::runners {

namespace {

// The mock's own shape. These are not config: nothing else in the server needs
// them, and the mock is a plumbing stand-in rather than a stand-in for a
// specific model. They match BGE-large so responses look plausible to clients.
constexpr size_t kEmbeddingDim = 1024;
constexpr size_t kMaxSequenceLength = 384;

// FNV-1a. Spelled out rather than using std::hash so the vectors are
// reproducible across standard-library implementations and machines - tests
// compare mock output against values captured elsewhere.
uint64_t hashInput(const std::string& text) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : text) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

// xorshift64*, seeded from the input hash: cheap, deterministic, and spreads
// bits well enough that different prompts give clearly different vectors.
uint64_t nextRandom(uint64_t& state) {
  state ^= state >> 12;
  state ^= state << 25;
  state ^= state >> 27;
  return state * 2685821657736338717ull;
}

std::vector<float> deterministicVector(const std::string& text, size_t dim) {
  uint64_t state = hashInput(text) | 1ull;  // xorshift needs a non-zero seed
  std::vector<float> v(dim);
  double sumSquares = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    // Map to [-1, 1).
    const auto bits = static_cast<uint32_t>(nextRandom(state) >> 32);
    const double value = static_cast<double>(bits) / 2147483648.0 - 1.0;
    v[i] = static_cast<float>(value);
    sumSquares += value * value;
  }
  // Normalize to unit length so the output has the shape callers expect from
  // an embedding (cosine comparisons behave sensibly).
  const double norm = std::sqrt(sumSquares);
  if (norm > 0.0) {
    for (auto& x : v) x = static_cast<float>(x / norm);
  }
  return v;
}

// Rough stand-in for the tokenizer: whitespace-separated words plus the two
// special tokens a BERT-style model adds, clamped to the model's limit so the
// truncation behaviour of the real path is visible in the mock too.
int approximateTokenCount(const std::string& text, size_t maxSequenceLength) {
  size_t words = 0;
  bool inWord = false;
  for (char c : text) {
    const bool space = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
    if (!space && !inWord) {
      ++words;
      inWord = true;
    } else if (space) {
      inWord = false;
    }
  }
  const size_t withSpecials = words + 2;
  return static_cast<int>(maxSequenceLength > 0
                              ? std::min(withSpecials, maxSequenceLength)
                              : withSpecials);
}

}  // namespace

MockEmbeddingRunner::MockEmbeddingRunner(const config::EmbeddingConfig& config)
    : config_(config) {
  TT_LOG_INFO(
      "[MockEmbeddingRunner] Created for model {} (dim={}, max_batch_size={}) "
      "- no Python, no device",
      config_.hf_model_id, kEmbeddingDim, config_.max_batch_size);
}

bool MockEmbeddingRunner::warmup() {
  TT_LOG_INFO("[MockEmbeddingRunner] Warmup complete (nothing to load)");
  return true;
}

std::vector<domain::EmbeddingResponse> MockEmbeddingRunner::run(
    const std::vector<domain::EmbeddingRequest>& requests) {
  std::vector<domain::EmbeddingResponse> responses;
  responses.reserve(requests.size());

  // Mirror the real runner's failure modes so the mock can catch regressions
  // in the layers above it: an oversized batch and an unknown model name both
  // raise on the Python side.
  if (requests.size() > config_.max_batch_size) {
    const std::string error = "Batch size " + std::to_string(requests.size()) +
                              " exceeds max " +
                              std::to_string(config_.max_batch_size);
    TT_LOG_ERROR("[MockEmbeddingRunner] {}", error);
    for (const auto& req : requests) {
      domain::EmbeddingResponse resp(req.task_id);
      resp.error = error;
      responses.push_back(std::move(resp));
    }
    return responses;
  }

  for (const auto& req : requests) {
    domain::EmbeddingResponse resp(req.task_id);
    resp.model = req.model;
    if (req.model != config_.hf_model_id) {
      resp.error = "Only " + config_.hf_model_id + " embeddings are supported";
      responses.push_back(std::move(resp));
      continue;
    }
    resp.embedding = deterministicVector(req.input, kEmbeddingDim);
    resp.total_tokens = approximateTokenCount(req.input, kMaxSequenceLength);
    responses.push_back(std::move(resp));
  }

  TT_LOG_DEBUG("[MockEmbeddingRunner] Answered {} requests", responses.size());
  return responses;
}

void MockEmbeddingRunner::close() {}

}  // namespace tt::runners
