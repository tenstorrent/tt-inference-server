// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include <unordered_map>

#include "profiling/tracy.hpp"
#include "runners/llm_runner/model_runner.hpp"
#include "utils/logger.hpp"

namespace tt::runners::llm_engine {

using Sequence = tt::domain::Sequence;
using TokenResult = tt::domain::TokenResult;
using Config = tt::config::LLMConfig;

namespace {

constexpr int64_t K_WHITESPACE_TOKEN_ID = 223;

// DeepSeek-R1-0528 tokenizer single-character token IDs (stable).
// Digits 0-9 occupy IDs 18-27; single ASCII letters follow sequentially.
constexpr int K_QUOTE_TOKEN = 4;         // '"'
constexpr int K_LETTER_A_TOKEN = 35;     // 'A' — only valid inside JSON string values
constexpr int K_FIRST_DIGIT_TOKEN = 18;  // '0'
constexpr int K_NUM_DIGITS = 10;
constexpr int K_MINUS_TOKEN = 15;        // '-'
constexpr int K_COMMA_TOKEN = 14;          // ','
constexpr int K_CLOSE_BRACKET_TOKEN = 63;  // ']'
constexpr int K_CLOSE_BRACE_TOKEN = 95;    // '}'

// Token IDs spelling "TT-Inference-Server" (hyphens substituted for spaces).
// Emitted length scales with max_tokens budget and varies per taskId.
constexpr std::array<int, 19> K_MOCK_STRING_CHARS = {
    54, 54, 15, 43, 80, 72, 71, 84, 71, 80, 69, 71, 15, 53, 71, 84, 88, 71, 84};

class MockModelRunner : public IModelRunner {
 public:
  MockModelRunner(const Config& config, DecodeCallback callback)
      : config(config), decodeCallback(std::move(callback)) {}

  void run(const std::vector<Sequence*>& seqs, bool isPrefill) override {
    ZoneScopedN("MockModelRunner::run");
    TT_LOG_DEBUG("[model_runner:mock] {} max_in_flight_count={}",
                 isPrefill ? "prefill" : "decode", seqs.size());
    if (isPrefill) {
      ZoneScopedN("MockModelRunner::prefill");
      for (Sequence* seq : seqs) {
        uint64_t tokenId = pickToken(seq, K_WHITESPACE_TOKEN_ID);
        decodeCallback(TokenResult(seq->taskId, tokenId));
      }
    } else {
      ZoneScopedN("MockModelRunner::decode");
      for (Sequence* seq : seqs) {
        uint64_t defaultToken = static_cast<uint64_t>(seq->getLastToken() + 1);
        uint64_t tokenId = pickToken(seq, defaultToken);
        decodeCallback(TokenResult(seq->taskId, tokenId));
      }
    }
  }

  void exit() override { TT_LOG_DEBUG("[model_runner:mock] exit"); }

 private:
  static bool isBitmaskSet(const std::vector<int32_t>& bitmask, int tokenId) {
    if (tokenId < 0) return false;
    size_t word = static_cast<size_t>(tokenId) / 32;
    if (word >= bitmask.size()) return false;
    return (static_cast<uint32_t>(bitmask[word]) >> (tokenId % 32)) & 1;
  }

  uint64_t pickToken(const Sequence* seq, uint64_t defaultToken) const {
    const auto& bitmask = seq->getSamplingParams().token_bitmask;
    if (bitmask.has_value()) {
      // 'A' being valid signals we are inside a JSON string value.
      if (isBitmaskSet(*bitmask, K_LETTER_A_TOKEN)) {
        int& count = stringCharCounts[seq->taskId];
        const auto& maxTok = seq->getSamplingParams().max_tokens;
        int budget = maxTok.has_value() ? *maxTok : 10;
        int baseChars = std::max(1, budget / 10);
        int maxChars = std::min(
            static_cast<int>(K_MOCK_STRING_CHARS.size()),
            baseChars + static_cast<int>(seq->taskId % 3));
        if (count < maxChars) {
          int charToken =
              K_MOCK_STRING_CHARS[count % K_MOCK_STRING_CHARS.size()];
          if (isBitmaskSet(*bitmask, charToken)) {
            ++count;
            return static_cast<uint64_t>(charToken);
          }
        }
        count = 0;  // Reset for the next string field in the same request.
        if (isBitmaskSet(*bitmask, K_QUOTE_TOKEN)) return K_QUOTE_TOKEN;
      }

      // Pick the lowest valid bitmask token with three overrides:
      //   ',' + ']' → prefer ']' to close arrays after one item;
      //   digit + '}'/']' → prefer the closing token to avoid integer repetition;
      //   '-' → substitute a task-varied digit (positive integers, varied output).
      for (size_t w = 0; w < bitmask->size(); ++w) {
        auto word = static_cast<uint32_t>((*bitmask)[w]);
        if (word != 0) {
          auto firstBit =
              static_cast<uint64_t>(w * 32 + __builtin_ctz(word));
          if (firstBit == K_COMMA_TOKEN &&
              isBitmaskSet(*bitmask, K_CLOSE_BRACKET_TOKEN)) {
            return K_CLOSE_BRACKET_TOKEN;
          }
          const bool isDigit = (firstBit >= K_FIRST_DIGIT_TOKEN &&
                                 firstBit < K_FIRST_DIGIT_TOKEN + K_NUM_DIGITS);
          if (isDigit) {
            if (isBitmaskSet(*bitmask, K_CLOSE_BRACE_TOKEN))
              return K_CLOSE_BRACE_TOKEN;
            if (isBitmaskSet(*bitmask, K_CLOSE_BRACKET_TOKEN))
              return K_CLOSE_BRACKET_TOKEN;
          }
          if (firstBit == K_MINUS_TOKEN) {
            int digit = K_FIRST_DIGIT_TOKEN +
                        static_cast<int>(seq->taskId % K_NUM_DIGITS);
            if (isBitmaskSet(*bitmask, digit)) return static_cast<uint64_t>(digit);
          }
          return firstBit;
        }
      }
      return defaultToken;
    }
    const auto& allowed = seq->getSamplingParams().allowed_token_ids;
    if (!allowed.has_value() || allowed->empty()) return defaultToken;

    int target = static_cast<int>(defaultToken);
    for (int id : *allowed) {
      if (id == target) return defaultToken;
    }
    return static_cast<uint64_t>(allowed->front());
  }

  Config config;
  DecodeCallback decodeCallback;
  // Per-task string char count; mutable because pickToken is const.
  mutable std::unordered_map<uint32_t, int> stringCharCounts;
};

}  // namespace

std::unique_ptr<IModelRunner> makeMockModelRunner(const Config& config,
                                                  DecodeCallback callback) {
  return std::make_unique<MockModelRunner>(config, std::move(callback));
}

}  // namespace tt::runners::llm_engine
