// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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
 * or a decode-only batch (no mixed batches). Subclasses control whether prefill
 * or decode is attempted first via should_prefill_first().
 */
class Scheduler {
 public:
  explicit Scheduler(const Config& config, ITaskQueue* task_queue);
  virtual ~Scheduler() = default;

  bool is_finished() const;

  Sequence& add_request(std::vector<int64_t> prompt,
                        const SamplingParams& params = SamplingParams());

  void add(Sequence& seq);

  Sequence* find_sequence(TaskID task_id);

  /**
   * Produces the next batch to run.
   * @return Pair of (scheduled sequences, is_prefill). The batch is either
   *         prefill-only or decode-only; should_prefill_first() determines
   *         which is attempted first.
   */
  std::pair<std::vector<Sequence*>, bool> schedule();

  void preempt(Sequence& seq);

  void postprocess(std::vector<Sequence*>& seqs,
                   const std::vector<int64_t>& token_ids);
  void removeSequence(TaskID task_id);

  bool is_stop_token(int64_t token_id) const { return stop_token_ids_.count(token_id) > 0; }

 protected:
  /**
   * @param has_waiting  true when the waiting queue has at least one request.
   * @param running_count  number of sequences currently in the running queue.
   * @param max_num_seqs  maximum batch / running capacity.
   * @return true if the scheduler should attempt prefill before decode.
   */
  virtual bool should_prefill_first(bool has_waiting, int running_count,
                                    int max_num_seqs) const = 0;

 private:
  int block_size_;
  bool try_schedule_prefill(std::vector<Sequence*>& scheduled_seqs,
                            int& num_seqs, int& num_batched_tokens);
  void try_schedule_decode(std::vector<Sequence*>& scheduled_seqs,
                           int& num_seqs);

  int max_num_seqs_;
  int max_num_batched_tokens_;
  std::unordered_set<int64_t> stop_token_ids_;
  BlockManager block_manager_;
  ITaskQueue* waiting_;
  std::unordered_map<TaskID, std::unique_ptr<Sequence>> sequences_;
  std::deque<Sequence*> running_;
  int in_flight_count_ = 0;
};

std::unique_ptr<Scheduler> make_scheduler(const Config& config,
                                          ITaskQueue* task_queue);

}  // namespace llm_engine
