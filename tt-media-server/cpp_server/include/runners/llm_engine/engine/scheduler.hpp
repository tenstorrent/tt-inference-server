#pragma once

#include <deque>
#include <vector>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/block_manager.hpp"
#include "llm_engine/engine/sequence.hpp"

namespace llm_engine {

/**
 * Schedules prefill and decode batches. Each step returns either a prefill-only
 * or a decode-only batch (no mixed batches). Prefill is prioritized over decode.
 */
class Scheduler {
 public:
  explicit Scheduler(const Config& config);

  /** @return true if there are no waiting, running, or in-flight sequences. */
  bool is_finished() const;

  /** Enqueues a sequence for prefill (waiting queue). */
  void add(Sequence& seq);

  /**
   * Produces the next batch to run.
   * @return Pair of (scheduled sequences, is_prefill). The batch is either
   *         prefill-only or decode-only; prefill is chosen when any waiting
   *         sequences can be scheduled.
   */
  std::pair<std::vector<Sequence*>, bool> schedule();

  /**
   * Moves a sequence from running back to waiting and frees its KV cache blocks.
   */
  void preempt(Sequence& seq);

  /**
   * Appends the generated token to each sequence and marks finished / deallocates
   * as needed. Call after the model runner returns token_ids for the batch.
   * @param seqs  The batch that was just run (same order as token_ids).
   * @param token_ids  One token per sequence from the model.
   */
  void postprocess(std::vector<Sequence*>& seqs,
                   const std::vector<int64_t>& token_ids);

 private:
  int max_num_seqs_;
  int max_num_batched_tokens_;
  int eos_;
  BlockManager block_manager_;
  std::deque<Sequence*> waiting_;
  std::deque<Sequence*> running_;
  int in_flight_count_ = 0;
};

}  // namespace llm_engine
