// SPDX-License-Identifier: Apache-2.0
#include "utils/id_generator.hpp"
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Smoke test: two processes communicate through a Boost IPC task queue.
//
//   Parent  -- creates the shared-memory message queue, pushes two sequences,
//              then forks.
//   Child   -- opens the queue via BoostIpcTaskQueue, creates a Scheduler,
//              calls schedule(), and verifies the deserialized batch.
//
// Exit code 0 = PASS.
#include <sys/wait.h>
#include <unistd.h>

#include <boost/interprocess/ipc/message_queue.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "config/runner_config.hpp"
#include "domain/sampling_params.hpp"
#include "domain/sequence.hpp"
#include "ipc/boost_ipc_task_queue.hpp"
#include "runners/schedulers/prefill_first_scheduler.hpp"
#include "runners/schedulers/scheduler.hpp"

using Sequence = tt::domain::Sequence;
using SamplingParams = tt::domain::SamplingParams;
namespace ipc = boost::interprocess;

static const char* queueName = "tt_ipc_scheduler_smoke_test";
static constexpr size_t MAX_NUM_MSGS = 64;
static constexpr size_t MAX_MSG_SIZE = 4096;

int main() {
  using namespace tt::runners::schedulers;
  using Config = tt::config::LLMConfig;

  // Clean up any leftover queue from a previous failed run.
  ipc::message_queue::remove(queueName);

  // --- Create the IPC queue (simulates the main server). ---
  ipc::message_queue rawQueue(ipc::create_only, queueName, MAX_NUM_MSGS,
                              MAX_MSG_SIZE);

  // Build two sequences with known values.
  uint32_t seq1Id = tt::utils::TaskIDGenerator::generate();
  uint32_t seq2Id = tt::utils::TaskIDGenerator::generate();
  Sequence seq1(seq1Id, 256, {1, 2, 3, 4}, SamplingParams{.max_tokens = 10});
  Sequence seq2(seq2Id, 256, {10, 20, 30},
                SamplingParams{.temperature = 0.7f, .max_tokens = 5});

  // Push via BoostIpcTaskQueue (opens the existing shared-memory queue).
  {
    tt::ipc::BoostIpcTaskQueue producer(queueName);
    producer.push(seq1);
    producer.push(seq2);
  }

  std::cout << "[parent] pushed task_id=" << seq1Id
            << " tokens=[1,2,3,4] max_tokens=10\n";
  std::cout << "[parent] pushed task_id=" << seq2Id
            << " tokens=[10,20,30] max_tokens=5 temperature=0.7\n";

  // --- Fork ---
  pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    ipc::message_queue::remove(queueName);
    return 1;
  }

  if (pid == 0) {
    // ---- Child process: read from the queue via Scheduler ----
    Config config;
    config.num_kvcache_blocks = 32;
    config.kvcache_block_size = 8;
    config.max_num_batched_tokens = 256;
    config.eos = 0;

    auto queue = std::make_unique<tt::ipc::BoostIpcTaskQueue>(queueName);
    PrefillFirstScheduler sched(config, queue.get(), 1);

    auto [batch, is_prefill] = sched.schedule();

    std::cout << "[child]  schedule() returned " << batch.size()
              << " sequences, is_prefill=" << is_prefill << "\n";

    for (auto* s : batch) {
      std::cout << "[child]    task_id=" << s->taskId << " size=" << s->size()
                << " max_tokens="
                << (s->getSamplingParams().max_tokens.has_value()
                        ? std::to_string(
                              s->getSamplingParams().max_tokens.value())
                        : "none")
                << " temperature=" << s->getSamplingParams().temperature
                << " tokens=[";
      for (size_t i = 0; i < s->size(); ++i) {
        if (i > 0) std::cout << ",";
        std::cout << (*s)[i];
      }
      std::cout << "]\n";
    }

    // --- Verify (batch size is 1 with current max_in_flight_count setting) ---
    bool ok = true;
    auto fail = [&](const char* msg) {
      std::cerr << "[child]  FAIL: " << msg << "\n";
      ok = false;
    };

    if (batch.size() != 1) fail("expected 1 sequence in batch");
    if (!is_prefill) fail("expected prefill batch");

    if (ok) {
      if (batch[0]->taskId != seq1Id) fail("seq1 task_id mismatch");
      if (batch[0]->size() != 4) fail("seq1 size mismatch");
      if (batch[0]->getSamplingParams().max_tokens != 10)
        fail("seq1 max_tokens mismatch");
      if ((*batch[0])[0] != 1 || (*batch[0])[1] != 2 || (*batch[0])[2] != 3 ||
          (*batch[0])[3] != 4)
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
  ipc::message_queue::remove(queueName);

  if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
    std::cout << "[parent] PASS: IPC scheduler smoke test\n";
    return 0;
  }
  std::cerr << "[parent] FAIL: child exited with status " << WEXITSTATUS(status)
            << "\n";
  return 1;
}
