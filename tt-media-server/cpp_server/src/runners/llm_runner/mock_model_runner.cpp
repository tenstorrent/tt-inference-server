// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "profiling/tracy.hpp"
#include "runners/llm_runner/model_runner.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::runners::llm_engine {

using Sequence = tt::domain::Sequence;
using TokenResult = tt::domain::TokenResult;
using Config = tt::config::LLMConfig;

namespace {

constexpr int64_t K_WHITESPACE_TOKEN_ID = 223;

// Token IDs for structural JSON characters and mock string content, resolved
// from the active tokenizer at construction time so the mock works with any
// vocabulary (DeepSeek, Llama, etc.).
struct GrammarTokenIds {
  int quote;          // '"'
  int letterA;        // 'A' — valid in bitmask only inside free-form string values
  int minus;          // '-'
  int comma;          // ','
  int closeBracket;   // ']'
  int closeBrace;     // '}'
  std::array<int, 10> digits;        // '0'–'9' (not assumed consecutive)
  std::array<int, 5> mockStrChars;   // T I S R V — varied content per task

  static GrammarTokenIds fromTokenizer(
      const tt::utils::tokenizers::Tokenizer& tok) {
    auto id = [&](char c) -> int {
      auto ids = tok.encode(std::string(1, c));
      if (ids.size() != 1)
        TT_LOG_WARN("[model_runner:mock] '{}' encodes to {} tokens, expected 1",
                    c, ids.size());
      return ids.empty() ? -1 : ids[0];
    };
    return {
        .quote = id('"'),
        .letterA = id('A'),
        .minus = id('-'),
        .comma = id(','),
        .closeBracket = id(']'),
        .closeBrace = id('}'),
        .digits = {id('0'), id('1'), id('2'), id('3'), id('4'),
                   id('5'), id('6'), id('7'), id('8'), id('9')},
        .mockStrChars = {id('T'), id('I'), id('S'), id('R'), id('V')},
    };
  }

  bool isDigit(int tokenId) const {
    for (int d : digits) {
      if (d == tokenId) return true;
    }
    return false;
  }
};

class MockModelRunner : public IModelRunner {
 public:
  MockModelRunner(const Config& config, DecodeCallback callback)
      : config(config),
        decodeCallback(std::move(callback)),
        tokenIds(GrammarTokenIds::fromTokenizer(
            tt::utils::tokenizers::activeTokenizer())) {}

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
  uint64_t pickToken(const Sequence* seq, uint64_t defaultToken) const {
    const auto& sp = seq->getSamplingParams();
    const auto& bitmask = sp.token_bitmask;
    if (bitmask.has_value()) {
      if (isBitmaskSet(*bitmask, tokenIds.letterA)) {
        // If the previous token was not '"' (opening quote), we already emitted
        // one content char → close the string now.
        if (seq->getLastToken() != static_cast<int64_t>(tokenIds.quote))
          return tokenIds.quote;
        // Emit one task-varied char from the mock string.
        int charToken = tokenIds.mockStrChars[seq->taskId % tokenIds.mockStrChars.size()];
        return isBitmaskSet(*bitmask, charToken) ? static_cast<uint64_t>(charToken)
                                                 : tokenIds.quote;
      }

      for (size_t w = 0; w < bitmask->size(); ++w) {
        auto word = static_cast<uint32_t>((*bitmask)[w]);
        if (word == 0) continue;
        const int candidateToken = static_cast<int>(w * 32 + __builtin_ctz(word));
        if (candidateToken == tokenIds.comma &&
            isBitmaskSet(*bitmask, tokenIds.closeBracket)) {
          return tokenIds.closeBracket;
        }
        if (tokenIds.isDigit(candidateToken)) {
          if (isBitmaskSet(*bitmask, tokenIds.closeBrace)) return tokenIds.closeBrace;
          if (isBitmaskSet(*bitmask, tokenIds.closeBracket)) return tokenIds.closeBracket;
        }
        if (candidateToken == tokenIds.minus) {
          int digit = tokenIds.digits[seq->taskId % tokenIds.digits.size()];
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
  GrammarTokenIds tokenIds;
};

}  // namespace

std::unique_ptr<IModelRunner> makeMockModelRunner(const Config& config,
                                                  DecodeCallback callback) {
  return std::make_unique<MockModelRunner>(config, std::move(callback));
}

}  // namespace tt::runners::llm_engine
