// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "llm_engine/engine/sequence.hpp"

namespace llm_engine {

/// Maximum number of tokens a single sequence can carry through the IPC queue.
static constexpr size_t MAX_SEQUENCE_TOKENS = 16384;

/**
 * Fixed-layout message for transferring Sequence data through a Boost IPC
 * message queue.  The token_ids array is sized to MAX_SEQUENCE_TOKENS; only
 * the first `num_tokens` entries are meaningful.
 */
struct SequenceMessage {
  int seq_id;
  float temperature;
  int max_tokens;
  bool ignore_eos;
  int64_t last_token;
  size_t num_prompt_tokens;
  size_t num_cached_tokens;
  size_t num_tokens;
  int64_t token_ids[MAX_SEQUENCE_TOKENS];
};

/// Serialize a Sequence into a SequenceMessage for IPC transfer.
inline SequenceMessage to_sequence_message(const Sequence& seq) {
  assert(seq.size() <= MAX_SEQUENCE_TOKENS);
  SequenceMessage msg{};
  msg.seq_id = seq.seq_id;
  msg.temperature = seq.temperature;
  msg.max_tokens = seq.max_tokens;
  msg.ignore_eos = seq.ignore_eos;
  msg.last_token = seq.last_token;
  msg.num_prompt_tokens = seq.num_prompt_tokens_;
  msg.num_cached_tokens = seq.num_cached_tokens_;
  msg.num_tokens = seq.size();
  std::memcpy(msg.token_ids, seq.token_ids_.data(),
              msg.num_tokens * sizeof(int64_t));
  return msg;
}

/// Deserialize a SequenceMessage into a heap-allocated Sequence.
/// Caller takes ownership of the returned pointer.
inline Sequence* from_sequence_message(const SequenceMessage& msg) {
  std::vector<int64_t> token_ids(msg.token_ids,
                                 msg.token_ids + msg.num_tokens);
  return new Sequence(msg.seq_id, std::move(token_ids), msg.num_prompt_tokens,
                      msg.num_cached_tokens, msg.temperature, msg.max_tokens,
                      msg.ignore_eos);
}

}  // namespace llm_engine
