#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "llm_engine/config.hpp"
#include "llm_engine/engine/block_manager.hpp"
#include "llm_engine/engine/sequence.hpp"
#include "llm_engine/engine/task_queue.hpp"

namespace llm_engine {

/**
 * Schedules prefill and decode batches. Each step returns either a prefill-only
 * or a decode-only batch (no mixed batches). Prefill is prioritized over decode.
 *
 * The waiting queue is an ITaskQueue (e.g. backed by Boost IPC message queue).
 * Multiple schedulers (across worker processes) can pop from the same queue.
 * The running queue is local to each scheduler instance.
 */
class Scheduler {
 public:
  explicit Scheduler(const Config& config);

  /** @return true if the task queue is empty and no sequences are running. */
  bool is_finished() const;

  /** Pushes a sequence to the task queue. */
  void add(Sequence& seq);

  /**
   * Produces the next batch to run.
   *
   * Prefill path: pops sequences from the task queue and returns raw pointers.
   * The caller is responsible for taking ownership of prefill sequences.
   *
   * Decode path: returns raw pointers to sequences already in running_.
   *
   * @return Pair of (scheduled sequences, is_prefill).
   */
  std::pair<std::vector<Sequence*>, bool> schedule();

  /**
   * Pushes the sequence back to the task queue and removes it from
   * running_.  Frees its KV cache blocks.
   */
  void preempt(Sequence& seq);

  /**
   * Appends the generated token to each sequence and marks finished /
   * deallocates as needed.
   */
  void postprocess(std::vector<Sequence*>& seqs,
                   const std::vector<int64_t>& token_ids);

 private:
  int max_num_seqs_;
  int max_num_batched_tokens_;
  int eos_;
  BlockManager block_manager_;
  std::unique_ptr<ITaskQueue> task_queue_;
  std::deque<Sequence*> running_;
};

}  // namespace llm_engine
