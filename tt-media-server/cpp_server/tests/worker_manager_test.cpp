// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Tests for the worker liveness logic exercised by WorkerManager.
//
// We test the pure helper `tt::worker::pollProcessLiveness()` rather than
// driving a real WorkerManager instance, because spinning up worker
// subprocesses requires the full server stack.  The helper isolates the
// waitpid/transition bookkeeping that the death callback relies on.

#include "runtime/worker/worker_manager.hpp"

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <thread>

namespace {

pid_t forkAndWaitForChildToExit() {
  pid_t pid = fork();
  if (pid == 0) {
    _exit(0);
  }
  // _exit() after fork is essentially instant; this sleep just gives the
  // kernel time to reparent the now-zombie child to us. We deliberately do
  // not waitpid() — pollProcessLiveness() must be the one to reap.
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  return pid;
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
  pid_t childPid = forkAndWaitForChildToExit();
  ASSERT_GT(childPid, 0) << "fork() failed";

  std::atomic<bool> alive{true};

  auto first =
      tt::worker::pollProcessLiveness(childPid, alive, /*workerIdx=*/3);
  EXPECT_FALSE(first.stillAlive);
  EXPECT_TRUE(first.transitionedToDead);
  EXPECT_FALSE(alive.load());

  // Second call: child has been reaped; the helper must not claim another
  // transition and stillAlive must mirror the (now false) aliveFlag.
  auto second =
      tt::worker::pollProcessLiveness(childPid, alive, /*workerIdx=*/3);
  EXPECT_FALSE(second.transitionedToDead);
  EXPECT_FALSE(second.stillAlive);
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
  pid_t childPid = forkAndWaitForChildToExit();
  ASSERT_GT(childPid, 0) << "fork() failed";

  std::atomic<bool> alive{false};  // Caller has already observed the death.

  auto result =
      tt::worker::pollProcessLiveness(childPid, alive, /*workerIdx=*/0);

  EXPECT_FALSE(result.stillAlive);
  EXPECT_FALSE(result.transitionedToDead)
      << "Transition should fire only once across all observers";
}
