// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "domain/session.hpp"

#include <cstdio>
#include <mutex>
#include <random>

#include "utils/block_hash_accumulator.hpp"

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

  // Finalize any partial block and register the final hash vector
  if (prefixAccumulator_ && onBlockComplete_) {
    auto finalHashes = prefixAccumulator_->finalize();
    if (!finalHashes.empty()) {
      onBlockComplete_(session_id_, finalHashes);
    }
  }

  prefixAccumulator_.reset();
  onBlockComplete_ = nullptr;
  return true;
}

void Session::initPrefixAccumulator(
    std::vector<uint64_t> initialHashes,
    std::vector<int64_t> partialBlockTokens,
    std::function<void(const std::string&, const std::vector<uint64_t>&)>
        onBlockComplete) {
  prefixAccumulator_ = std::make_shared<utils::BlockHashAccumulator>(
      std::move(initialHashes), std::move(partialBlockTokens));
  onBlockComplete_ = std::move(onBlockComplete);
}

void Session::addGeneratedToken(int64_t tokenId) {
  if (!prefixAccumulator_) return;

  if (auto newHashes = prefixAccumulator_->addToken(tokenId)) {
    if (onBlockComplete_) {
      onBlockComplete_(session_id_, *newHashes);
    }
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
