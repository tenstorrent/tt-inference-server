// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

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
constexpr int K_QUOTE_TOKEN = 4;           // '"'
constexpr int K_LETTER_A_TOKEN = 35;       // 'A' — valid only inside string values
constexpr int K_FIRST_DIGIT_TOKEN = 18;    // '0'
constexpr int K_NUM_DIGITS = 10;
constexpr int K_MINUS_TOKEN = 15;          // '-'
constexpr int K_COMMA_TOKEN = 14;          // ','
constexpr int K_CLOSE_BRACKET_TOKEN = 63;  // ']'
constexpr int K_CLOSE_BRACE_TOKEN = 95;    // '}'

// A small pool of distinct letter tokens used to vary mock string content per task.
// T(54) I(43) S(53) R(84) V(88)
constexpr std::array<int, 5> K_MOCK_STRING_CHARS = {54, 43, 53, 84, 88};

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
        decodeCallback(TokenResult(seq->taskId, pickToken(seq, K_WHITESPACE_TOKEN_ID)));
      }
    } else {
      ZoneScopedN("MockModelRunner::decode");
      for (Sequence* seq : seqs) {
        uint64_t defaultToken = static_cast<uint64_t>(seq->getLastToken() + 1);
        decodeCallback(TokenResult(seq->taskId, pickToken(seq, defaultToken)));
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

  // Picks the next token under grammar guidance.
  //
  // When a bitmask is present (structured output), three overrides apply:
  //   String value  — 'A' valid signals free-form string content; emit one
  //                   task-varied char then close on the following step, using
  //                   getLastToken() to detect which step we are on.
  //   Array close   — prefer ']' over ',' so arrays close after one item.
  //   Integer close — prefer '}'/']' over digit repetition once one digit
  //                   has been emitted (digit is lowest bit but close token
  //                   is also valid).
  //   '-'           — substitute a task-varied positive digit.
  static uint64_t pickToken(const Sequence* seq, uint64_t defaultToken) {
    const auto& sp = seq->getSamplingParams();
    const auto& bitmask = sp.token_bitmask;
    if (bitmask.has_value()) {
      if (isBitmaskSet(*bitmask, K_LETTER_A_TOKEN)) {
        // If the previous token was not '"' (opening quote), we already emitted
        // one content char → close the string now.
        if (seq->getLastToken() != static_cast<int64_t>(K_QUOTE_TOKEN))
          return K_QUOTE_TOKEN;
        // Emit one task-varied char from the mock string.
        int charToken = K_MOCK_STRING_CHARS[seq->taskId % K_MOCK_STRING_CHARS.size()];
        return isBitmaskSet(*bitmask, charToken) ? static_cast<uint64_t>(charToken)
                                                 : K_QUOTE_TOKEN;
      }

      for (size_t w = 0; w < bitmask->size(); ++w) {
        auto word = static_cast<uint32_t>((*bitmask)[w]);
        if (word == 0) continue;
        const int candidateToken = static_cast<int>(w * 32 + __builtin_ctz(word));
        if (candidateToken == K_COMMA_TOKEN &&
            isBitmaskSet(*bitmask, K_CLOSE_BRACKET_TOKEN)) {
          return K_CLOSE_BRACKET_TOKEN;
        }
        const bool isDigit = candidateToken >= K_FIRST_DIGIT_TOKEN &&
                             candidateToken < K_FIRST_DIGIT_TOKEN + K_NUM_DIGITS;
        if (isDigit) {
          if (isBitmaskSet(*bitmask, K_CLOSE_BRACE_TOKEN)) return K_CLOSE_BRACE_TOKEN;
          if (isBitmaskSet(*bitmask, K_CLOSE_BRACKET_TOKEN)) return K_CLOSE_BRACKET_TOKEN;
        }
        if (candidateToken == K_MINUS_TOKEN) {
          int digit = K_FIRST_DIGIT_TOKEN + static_cast<int>(seq->taskId % K_NUM_DIGITS);
          if (isBitmaskSet(*bitmask, digit)) return static_cast<uint64_t>(digit);
        }
        return static_cast<uint64_t>(candidateToken);
      }
      return defaultToken;
    }
    const auto& allowed = sp.allowed_token_ids;
    if (!allowed.has_value() || allowed->empty()) return defaultToken;
    int target = static_cast<int>(defaultToken);
    for (int id : *allowed) {
      if (id == target) return defaultToken;
    }
    return static_cast<uint64_t>(allowed->front());
  }

  Config config;
  DecodeCallback decodeCallback;
};

}  // namespace

std::unique_ptr<IModelRunner> makeMockModelRunner(const Config& config,
                                                  DecodeCallback callback) {
  return std::make_unique<MockModelRunner>(config, std::move(callback));
}

}  // namespace tt::runners::llm_engine
