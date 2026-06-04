// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/session.hpp"

#include <cstdio>
#include <mutex>
#include <random>

#include "utils/conversation_hasher.hpp"
#include "utils/logger.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::domain {

Session::Session(uint32_t slotId, size_t initialHash)
    : session_id_(generateUuid()),
      hash_(initialHash),
      slot_id_(slotId),
      last_activity_time_(std::chrono::system_clock::now()) {}

bool Session::markInFlight() {
  if (state_ != SessionState::PREPARED && state_ != SessionState::IDLE)
    return false;
  state_ = SessionState::IN_FLIGHT;
  return true;
}

bool Session::markPrepared() {
  if (state_ != SessionState::IDLE) return false;
  state_ = SessionState::PREPARED;
  return true;
}

bool Session::clearInFlight() {
  if (state_ != SessionState::IN_FLIGHT) return false;
  state_ = SessionState::IDLE;
  cancelFn_ = nullptr;
  deltaTokens_.clear();
  generatedTokens_.clear();
  initialBlocks_.clear();
  parentHash_ = 0;
  parentThinkCount_ = 0;
  onComplete_ = nullptr;
  inThinkingBlock_ = false;
  accumulatedThinkTokens_ = 0;
  return true;
}

void Session::initTokenAccumulator(
    std::vector<int> deltaTokens,
    std::vector<utils::BlockHashInfo> initialBlocks,
    std::function<void(const std::string&,
                       const std::vector<utils::BlockHashInfo>&)>
        onComplete) {
  deltaTokens_ = std::move(deltaTokens);
  initialBlocks_ = std::move(initialBlocks);
  parentHash_ = initialBlocks_.empty() ? 0 : initialBlocks_.back().hash;
  parentThinkCount_ =
      initialBlocks_.empty() ? 0 : initialBlocks_.back().accumulatedThinkTokens;
  onComplete_ = std::move(onComplete);
  generatedTokens_.clear();

  // Initialize thinking token tracking
  auto [thinkStart, thinkEnd] = utils::tokenizers::thinkTokenIds();
  thinkStartTokenId_ = thinkStart;
  thinkEndTokenId_ = thinkEnd;
  inThinkingBlock_ = false;
  accumulatedThinkTokens_ = parentThinkCount_;
}

void Session::addGeneratedToken(int tokenId) {
  generatedTokens_.push_back(tokenId);

  // Track thinking state using same state machine as ReasoningParser
  const bool thinkingEnabled =
      thinkStartTokenId_ != utils::tokenizers::kNoThinkTokenId &&
      thinkEndTokenId_ != utils::tokenizers::kNoThinkTokenId;
  if (!thinkingEnabled) return;

  if (tokenId == static_cast<int>(thinkStartTokenId_)) {
    inThinkingBlock_ = true;
  } else if (tokenId == static_cast<int>(thinkEndTokenId_)) {
    inThinkingBlock_ = false;
  } else if (inThinkingBlock_) {
    ++accumulatedThinkTokens_;  // Only content tokens, not markers
  }
}

void Session::finalizeAndRegisterHashes() {
  if (!onComplete_) return;

  // [EOS-DIAG] Confirm whether generation ran past the turn-end token
  // <|im_end|>. The blaze scheduler eos (commit 5d83c0b set it to 163585
  // [EOS]) must match the model's per-turn terminator (163586 <|im_end|>).
  // If <|im_end|> appears before the end of generatedTokens_, decode is
  // over-generating and the polluted tail gets registered into the prefix —
  // which breaks multi-turn prefix-cache reuse (only block 0 matches).
  // tokensAfterImEnd > 0  => over-generation (eos mismatch confirmed)
  // tokensAfterImEnd == 0 && firstImEndIdx >= 0 => clean stop at <|im_end|>
  // firstImEndIdx == -1   => <|im_end|> never emitted (stopped on EOS/length)
  // Remove after diagnosis.
  {
    const auto& stops = utils::tokenizers::staticInfo().stopTokenIds;
    const int64_t imEnd = stops.empty() ? -1 : stops.front();
    int firstImEndIdx = -1;
    for (size_t i = 0; i < generatedTokens_.size(); ++i) {
      if (generatedTokens_[i] == imEnd) {
        firstImEndIdx = static_cast<int>(i);
        break;
      }
    }
    const size_t tokensAfterImEnd =
        firstImEndIdx >= 0
            ? generatedTokens_.size() - 1 - static_cast<size_t>(firstImEndIdx)
            : 0;
    std::string tailIds;
    const size_t start =
        generatedTokens_.size() > 16 ? generatedTokens_.size() - 16 : 0;
    for (size_t i = start; i < generatedTokens_.size(); ++i) {
      tailIds += std::to_string(generatedTokens_[i]);
      tailIds += ' ';
    }
    TT_LOG_WARN(
        "[EOS-DIAG] session={} generatedTokens={} stopTok(im_end)={} "
        "firstImEndIdx={} tokensAfterImEnd={} lastTokenIds=[{}]",
        session_id_, generatedTokens_.size(), imEnd, firstImEndIdx,
        tokensAfterImEnd, tailIds);
  }

  // Combine delta prompt + generated tokens
  std::vector<int> allDeltaTokens = deltaTokens_;
  allDeltaTokens.insert(allDeltaTokens.end(), generatedTokens_.begin(),
                        generatedTokens_.end());

  // Compute new block info continuing from parent (avoids re-hashing matched
  // prefix). Uses thinking-aware hashing to exclude thinking tokens from hash.
  auto newBlocks = utils::getPrefixCacheHashesByBlocksWithThinking(
      allDeltaTokens, thinkStartTokenId_, thinkEndTokenId_, parentHash_,
      parentThinkCount_);

  // Only register if new blocks were formed
  if (!newBlocks.empty()) {
    // Prepend initial blocks to form complete block list
    std::vector<utils::BlockHashInfo> allBlocks = initialBlocks_;
    allBlocks.insert(allBlocks.end(), newBlocks.begin(), newBlocks.end());
    onComplete_(session_id_, allBlocks);
  }
}

std::string Session::generateUuid() {
  // Generate a stable UUID v4 for session identity
  static std::mutex genMutex;
  static std::mt19937_64 gen(std::random_device{}());

  std::lock_guard<std::mutex> lock(genMutex);
  uint64_t a = gen(), b = gen();

  a = (a & ~0xF000ULL) | 0x4000ULL;                         // version 4
  b = (b & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;  // variant 10xx

  char buf[37];
  snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%012llx",
           static_cast<uint32_t>(a >> 32),
           static_cast<uint32_t>((a >> 16) & 0xFFFF),
           static_cast<uint32_t>(a & 0xFFFF), static_cast<uint32_t>(b >> 48),
           static_cast<unsigned long long>(b & 0x0000FFFFFFFFFFFFULL));
  return buf;
}

}  // namespace tt::domain
