#pragma once

#include <deque>
#include <memory>
#include <unordered_set>
#include <vector>
#include <unordered_map>

#include "runners/llm_runner/config.hpp"
#include "runners/llm_runner/block_manager.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/task_queue.hpp"
#include "runners/llm_runner/sampling_params.hpp"

namespace llm_engine {

/**
 * Schedules prefill and decode batches. Each step returns either a prefill-only
 * or a decode-only batch (no mixed batches). Prefill is prioritized over decode.
 */
class Scheduler {
 public:
  explicit Scheduler(const Config& config, ITaskQueue* task_queue);

  /** @return true if there are no waiting, running, or in-flight sequences. */
  bool is_finished() const;

  /** Creates a sequence, takes ownership, and enqueues it for prefill. */
  Sequence& add_request(std::vector<int64_t> prompt,
                        const SamplingParams& params = SamplingParams());

  /** Enqueues an externally-owned sequence for prefill (waiting queue). */
  void add(Sequence& seq);

  /** Looks up a sequence by task_id. Returns nullptr if not found. */
  Sequence* find_sequence(TaskID task_id);

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
  void removeSequence(TaskID task_id);

  bool is_stop_token(int64_t token_id) const { return stop_token_ids_.count(token_id) > 0; }

 private:
  int max_num_seqs_;
  int max_num_batched_tokens_;
  std::unordered_set<int64_t> stop_token_ids_;
  BlockManager block_manager_;
  ITaskQueue* waiting_;
  std::unordered_map<TaskID, std::unique_ptr<Sequence>> sequences_;
  std::deque<Sequence*> running_;
  int in_flight_count_ = 0;
};

}  // namespace llm_engine
