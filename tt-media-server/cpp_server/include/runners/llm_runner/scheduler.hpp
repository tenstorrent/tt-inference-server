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
 * Decides whether prefill or decode should be attempted first in each
 * scheduling step. Implementations encode different scheduling policies.
 */
class ISchedulingStrategy {
 public:
  virtual ~ISchedulingStrategy() = default;

  /**
   * @param has_waiting  true when the waiting queue has at least one request.
   * @param running_count  number of sequences currently in the running queue.
   * @param max_num_seqs  maximum batch / running capacity.
   * @return true if the scheduler should attempt prefill before decode.
   */
  virtual bool should_prefill_first(bool has_waiting, int running_count,
                                    int max_num_seqs) = 0;

  /** Called after each scheduling step so the strategy can track state. */
  virtual void on_step_complete(bool was_prefill) = 0;
};

/**
 * Always prefills when the waiting queue has requests (original behaviour).
 * Decode is only attempted when nothing can be prefilled.
 */
class PrefillFirstStrategy : public ISchedulingStrategy {
 public:
  bool should_prefill_first(bool has_waiting, int /*running_count*/,
                            int /*max_num_seqs*/) override {
    return has_waiting;
  }
  void on_step_complete(bool /*was_prefill*/) override {}
};

/**
 * Interleaves prefill and decode steps.
 *
 * After a prefill step the strategy forces the next step to decode (if there
 * are running sequences). Running count is capped at max_num_seqs so that new
 * prefills are only admitted when running slots are available. This mirrors
 * the AscendScheduler / vLLM V1 approach where a batch is prefilled, decoded
 * to completion (or until slots free up), and only then is the next batch
 * admitted for prefill.
 */
class InterleavedStrategy : public ISchedulingStrategy {
 public:
  bool should_prefill_first(bool has_waiting, int running_count,
                            int max_num_seqs) override {
    if (!has_waiting) return false;
    if (running_count >= max_num_seqs) return false;
    if (running_count == 0) return true;
    return !last_was_prefill_;
  }
  void on_step_complete(bool was_prefill) override {
    last_was_prefill_ = was_prefill;
  }

 private:
  bool last_was_prefill_ = false;
};

std::unique_ptr<ISchedulingStrategy> make_scheduling_strategy(
    SchedulingPolicy policy);

/**
 * Schedules prefill and decode batches. Each step returns either a prefill-only
 * or a decode-only batch (no mixed batches). The ordering is controlled by the
 * SchedulingPolicy in Config (PREFILL_FIRST or INTERLEAVED).
 */
class Scheduler {
 public:
  explicit Scheduler(const Config& config, ITaskQueue* task_queue);

  bool is_finished() const;

  Sequence& add_request(std::vector<int64_t> prompt,
                        const SamplingParams& params = SamplingParams());

  void add(Sequence& seq);

  Sequence* find_sequence(TaskID task_id);

  /**
   * Produces the next batch to run.
   * @return Pair of (scheduled sequences, is_prefill). The batch is either
   *         prefill-only or decode-only; the ordering strategy determines
   *         which is attempted first.
   */
  std::pair<std::vector<Sequence*>, bool> schedule();

  void preempt(Sequence& seq);

  void postprocess(std::vector<Sequence*>& seqs,
                   const std::vector<int64_t>& token_ids);
  void removeSequence(TaskID task_id);

  bool is_stop_token(int64_t token_id) const { return stop_token_ids_.count(token_id) > 0; }

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
  std::unique_ptr<ISchedulingStrategy> strategy_;
};

}  // namespace llm_engine
