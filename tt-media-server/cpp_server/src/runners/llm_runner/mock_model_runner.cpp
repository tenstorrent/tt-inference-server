#include <optional>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "profiling/tracy.hpp"
#include "runners/llm_runner/model_runner.hpp"
#include "utils/logger.hpp"

namespace tt::runners::llm_engine {

using Sequence = tt::domain::Sequence;
using TokenResult = tt::domain::TokenResult;
using Config = tt::config::LLMConfig;

namespace {

constexpr int64_t K_WHITESPACE_TOKEN_ID = 223;

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
        uint64_t tokenId = pickTokenForSequence(seq);
        decodeCallback(TokenResult(seq->taskId, tokenId));
      }
    }
  }

  void exit() override { TT_LOG_DEBUG("[model_runner:mock] exit"); }

 private:
  uint64_t pickTokenForSequence(Sequence* seq) {
    uint64_t defaultToken = static_cast<uint64_t>(seq->getLastToken() + 1);
    uint32_t taskId = seq->taskId;

    const auto& bitmask = seq->getSamplingParams().token_bitmask;
    if (!bitmask.has_value()) {
      // No guided decoding - use simple logic
      return pickToken(seq, defaultToken);
    }

    int vocabSize = seq->getSamplingParams().bitmask_vocab_size;
    int target = static_cast<int>(defaultToken);
    bool defaultIsAllowed =
        (target < vocabSize && isBitmaskSet(*bitmask, target));

    // Track consecutive times we would have picked defaultToken
    if (defaultIsAllowed) {
      consecutiveDefaultPicks[taskId]++;

      // After 2 consecutive default picks, cycle through allowed tokens to find
      // terminators
      if (consecutiveDefaultPicks[taskId] >= 2) {
        consecutiveDefaultPicks[taskId] = 0;  // Reset counter

        // Collect all allowed tokens
        std::vector<uint64_t> allowed;
        for (size_t w = 0; w < bitmask->size(); ++w) {
          auto word = static_cast<uint32_t>((*bitmask)[w]);
          if (word != 0) {
            for (int bit = 0; bit < 32; ++bit) {
              if ((word >> bit) & 1) {
                allowed.push_back(w * 32 + bit);
              }
            }
          }
        }

        if (!allowed.empty()) {
          // Cycle through allowed tokens using an offset
          size_t& offset = tokenOffsets[taskId];
          uint64_t token = allowed[offset % allowed.size()];
          offset++;
          return token;
        }
      }

      return defaultToken;
    } else {
      // defaultToken not allowed, reset counter and pick lowest allowed
      consecutiveDefaultPicks[taskId] = 0;
      return pickToken(seq, defaultToken, true);
    }
  }

  static bool isBitmaskSet(const std::vector<int32_t>& bitmask, int tokenId) {
    return (static_cast<uint32_t>(bitmask[tokenId / 32]) >> (tokenId % 32)) & 1;
  }

  static uint64_t pickNonWhitespaceToken(const std::vector<int32_t>& bitmask) {
    // Whitespace tokens to avoid (tab, newline, carriage return, space)
    static const std::unordered_set<uint64_t> whitespace = {9, 10, 13, 32};
    // JSON structural tokens to prefer (helps terminate strings/objects)
    // " { } [ ] : ,
    static const std::unordered_set<uint64_t> structural = {34, 44,  58, 91,
                                                            93, 123, 125};

    std::optional<uint64_t> lowestStructural;
    std::optional<uint64_t> lowestNonWhitespace;
    std::optional<uint64_t> lowestWhitespace;

    for (size_t w = 0; w < bitmask.size(); ++w) {
      auto word = static_cast<uint32_t>(bitmask[w]);
      if (word != 0) {
        for (int bit = 0; bit < 32; ++bit) {
          if ((word >> bit) & 1) {
            uint64_t tokenId = w * 32 + bit;
            if (structural.count(tokenId)) {
              if (!lowestStructural.has_value() ||
                  tokenId < *lowestStructural) {
                lowestStructural = tokenId;
              }
            } else if (whitespace.count(tokenId)) {
              if (!lowestWhitespace.has_value() ||
                  tokenId < *lowestWhitespace) {
                lowestWhitespace = tokenId;
              }
            } else {
              if (!lowestNonWhitespace.has_value() ||
                  tokenId < *lowestNonWhitespace) {
                lowestNonWhitespace = tokenId;
              }
            }
          }
        }
      }
    }

    // Prefer structural tokens (terminators), then content, then whitespace
    if (lowestStructural.has_value()) {
      return *lowestStructural;
    } else if (lowestNonWhitespace.has_value()) {
      return *lowestNonWhitespace;
    } else if (lowestWhitespace.has_value()) {
      return *lowestWhitespace;
    }
    return 0;
  }

  static uint64_t pickToken(const Sequence* seq, uint64_t defaultToken,
                            bool forceRandom = false) {
    const auto& bitmask = seq->getSamplingParams().token_bitmask;
    if (bitmask.has_value()) {
      int vocabSize = seq->getSamplingParams().bitmask_vocab_size;

      // For guided decoding: pick lowest non-whitespace token to find
      // terminators
      if (forceRandom) {
        return pickNonWhitespaceToken(*bitmask);
      }

      // Normal path: try default token first
      int target = static_cast<int>(defaultToken);
      if (target < vocabSize && isBitmaskSet(*bitmask, target)) {
        return defaultToken;
      }

      // Fallback to first allowed token
      for (size_t w = 0; w < bitmask->size(); ++w) {
        auto word = static_cast<uint32_t>((*bitmask)[w]);
        if (word != 0) {
          return static_cast<uint64_t>(w * 32 + __builtin_ctz(word));
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
  std::unordered_map<uint32_t, int> consecutiveDefaultPicks;
  std::unordered_map<uint32_t, size_t> tokenOffsets;
};

}  // namespace

std::unique_ptr<IModelRunner> makeMockModelRunner(const Config& config,
                                                  DecodeCallback callback) {
  return std::make_unique<MockModelRunner>(config, std::move(callback));
}

}  // namespace tt::runners::llm_engine
