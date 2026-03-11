// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// Smoke test: two processes communicate through a Boost IPC task queue.
//
//   Parent  -- creates the shared-memory message queue, pushes two sequences,
//              then forks.
//   Child   -- opens the queue via BoostIpcTaskQueue, creates a Scheduler,
//              calls schedule(), and verifies the deserialized batch.
//
// Exit code 0 = PASS.
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include "runners/llm_runner/config.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "runners/llm_runner/scheduler.hpp"
#include "runners/llm_runner/prefill_first_scheduler.hpp"
#include "runners/llm_runner/sequence.hpp"
#include "runners/llm_runner/sampling_params.hpp"

#include <boost/interprocess/ipc/message_queue.hpp>

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

namespace ipc = boost::interprocess;

static const char* QUEUE_NAME = "tt_ipc_scheduler_smoke_test";
static constexpr size_t MAX_NUM_MSGS = 64;
static constexpr size_t MAX_MSG_SIZE = 4096;

int main() {
  using namespace llm_engine;

  // Clean up any leftover queue from a previous failed run.
  ipc::message_queue::remove(QUEUE_NAME);

  // --- Create the IPC queue (simulates the main server). ---
  ipc::message_queue raw_queue(
      ipc::create_only, QUEUE_NAME, MAX_NUM_MSGS, MAX_MSG_SIZE);

  // Build two sequences with known values.
  std::string seq1_id = boost::uuids::to_string(boost::uuids::random_generator()());
  std::string seq2_id = boost::uuids::to_string(boost::uuids::random_generator()());
  Sequence seq1(TaskID(seq1_id), 256, {1, 2, 3, 4}, SamplingParams{.max_tokens = 10});
  Sequence seq2(TaskID(seq2_id), 256, {10, 20, 30}, SamplingParams{.temperature = 0.7f, .max_tokens = 5});

  // Push via BoostIpcTaskQueue (opens the existing shared-memory queue).
  {
    tt::ipc::BoostIpcTaskQueue producer(QUEUE_NAME);
    producer.push(seq1);
    producer.push(seq2);
  }

  std::cout << "[parent] pushed task_id=" << seq1_id
            << " tokens=[1,2,3,4] max_tokens=10\n";
  std::cout << "[parent] pushed task_id=" << seq2_id
            << " tokens=[10,20,30] max_tokens=5 temperature=0.7\n";

  // --- Fork ---
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    ipc::message_queue::remove(QUEUE_NAME);
    return 1;
  }

  if (pid == 0) {
    // ---- Child process: read from the queue via Scheduler ----
    Config config;
    config.num_kvcache_blocks = 32;
    config.kvcache_block_size = 8;
    config.max_num_batched_tokens = 256;
    config.eos = 0;

    auto queue = std::make_unique<tt::ipc::BoostIpcTaskQueue>(QUEUE_NAME);
    PrefillFirstScheduler sched(config, queue.get());

    auto [batch, is_prefill] = sched.schedule();

    std::cout << "[child]  schedule() returned " << batch.size()
              << " sequences, is_prefill=" << is_prefill << "\n";

    for (auto* s : batch) {
      std::cout << "[child]    task_id=" << s->task_id
                << " size=" << s->size()
                << " max_tokens=" << s->sampling_params->max_tokens
                << " temperature=" << s->sampling_params->temperature
                << " tokens=[";
      for (size_t i = 0; i < s->size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << (*s)[i];
      }
      std::cout << "]\n";
    }

    // --- Verify (batch size is 1 with current Config::max_num_seqs) ---
    bool ok = true;
    auto fail = [&](const char* msg) {
      std::cerr << "[child]  FAIL: " << msg << "\n";
      ok = false;
    };

    if (batch.size() != 1) fail("expected 1 sequence in batch");
    if (!is_prefill) fail("expected prefill batch");

    if (ok) {
      if (batch[0]->task_id.id != seq1_id) fail("seq1 task_id mismatch");
      if (batch[0]->size() != 4) fail("seq1 size mismatch");
      if (batch[0]->sampling_params->max_tokens != 10) fail("seq1 max_tokens mismatch");
      if ((*batch[0])[0] != 1 || (*batch[0])[1] != 2 ||
          (*batch[0])[2] != 3 || (*batch[0])[3] != 4)
        fail("seq1 token values mismatch");
    }

    if (ok) std::cout << "[child]  PASS\n";
    std::cout.flush();
    std::cerr.flush();
    _exit(ok ? 0 : 1);
  }

  // --- Parent: wait for child, clean up. ---
  int status = 0;
  waitpid(pid, &status, 0);
  ipc::message_queue::remove(QUEUE_NAME);

  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    std::cout << "[parent] PASS: IPC scheduler smoke test\n";
    return 0;
  }
  std::cerr << "[parent] FAIL: child exited with status "
            << WEXITSTATUS(status) << "\n";
  return 1;
}
