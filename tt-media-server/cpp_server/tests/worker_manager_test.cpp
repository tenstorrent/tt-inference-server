// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Tests for the worker liveness logic exercised by WorkerManager.
//
// We test the pure helper `tt::worker::pollProcessLiveness()` rather than
// driving a real WorkerManager instance, because spinning up worker
// subprocesses requires the full server stack.  The helper isolates the
// waitpid/transition bookkeeping that the death callback relies on.

#include "worker/worker_manager.hpp"

#include <gtest/gtest.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace {

pid_t forkExitingChild() {
  pid_t pid = fork();
  if (pid == 0) {
    _exit(0);
  }
  return pid;
}

void waitForProcessExit(pid_t pid) {
  // We deliberately don't waitpid() here: pollProcessLiveness() must be the
  // one to reap the zombie so we can assert on the transition it returns.
  for (int i = 0; i < 200; ++i) {
    if (kill(pid, 0) != 0) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

}  // namespace

TEST(PollProcessLiveness, AliveProcessReportsAlive) {
  pid_t self = getpid();
  std::atomic<bool> alive{true};

  auto result = tt::worker::pollProcessLiveness(self, alive, /*workerIdx=*/0);

  EXPECT_TRUE(result.stillAlive);
  EXPECT_FALSE(result.transitionedToDead);
  EXPECT_TRUE(alive.load());
}

TEST(PollProcessLiveness, ExitedChildIsReapedAndTransitionsOnce) {
  pid_t childPid = forkExitingChild();
  ASSERT_GT(childPid, 0) << "fork() failed";
  waitForProcessExit(childPid);

  std::atomic<bool> alive{true};

  auto first =
      tt::worker::pollProcessLiveness(childPid, alive, /*workerIdx=*/3);
  EXPECT_FALSE(first.stillAlive);
  EXPECT_TRUE(first.transitionedToDead);
  EXPECT_FALSE(alive.load());

  // Second call: the child has already been reaped; the helper must not
  // claim a second transition.
  auto second =
      tt::worker::pollProcessLiveness(childPid, alive, /*workerIdx=*/3);
  EXPECT_FALSE(second.transitionedToDead);
}

TEST(PollProcessLiveness, InvalidPidTransitionsAtMostOnce) {
  std::atomic<bool> alive{true};

  auto first =
      tt::worker::pollProcessLiveness(/*pid=*/0, alive, /*workerIdx=*/0);
  EXPECT_FALSE(first.stillAlive);
  EXPECT_TRUE(first.transitionedToDead);

  auto second =
      tt::worker::pollProcessLiveness(/*pid=*/0, alive, /*workerIdx=*/0);
  EXPECT_FALSE(second.stillAlive);
  EXPECT_FALSE(second.transitionedToDead);
}

TEST(PollProcessLiveness, AlreadyDeadFlagDoesNotResurrectTransition) {
  pid_t childPid = forkExitingChild();
  ASSERT_GT(childPid, 0) << "fork() failed";
  waitForProcessExit(childPid);

  std::atomic<bool> alive{false};  // Caller has already observed the death.

  auto result =
      tt::worker::pollProcessLiveness(childPid, alive, /*workerIdx=*/0);

  EXPECT_FALSE(result.stillAlive);
  EXPECT_FALSE(result.transitionedToDead)
      << "Transition should fire only once across all observers";
}
