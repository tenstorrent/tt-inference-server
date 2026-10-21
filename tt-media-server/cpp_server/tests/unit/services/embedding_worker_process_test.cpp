// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// Unit tests for WorkerProcess (services/embedding_worker_process.cpp): the
// fork/pipe/waitpid lifecycle of one embedding worker, exercised through the
// childMain seam with trivial child lambdas instead of a real runner.

#include "services/embedding_worker_process.hpp"

#include <gtest/gtest.h>
#include <signal.h>
#include <unistd.h>

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "services/embedding_pipe.hpp"

namespace pipe_detail = tt::services::embedding_detail;
using pipe_detail::WorkerProcess;

namespace {

// Poll checkAlive until the child is reaped or the deadline passes; children
// need a moment between _exit and waitpid observing it.
bool waitUntilDead(WorkerProcess& worker, int timeoutMs = 2000) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
  while (std::chrono::steady_clock::now() < deadline) {
    if (!worker.checkAlive()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

// Child body: echo every length-prefixed message back until EOF.
void echoChildMain(int readFd, int writeFd) {
  while (true) {
    const std::string msg = pipe_detail::pipeReadString(readFd);
    if (msg.empty()) break;
    pipe_detail::pipeWrite(writeFd, msg.data(), msg.size());
  }
  _exit(0);
}

class EmbeddingWorkerProcessTest : public ::testing::Test {
 protected:
  // A dead child's pipe raises SIGPIPE on write; the parent must survive it.
  void SetUp() override { signal(SIGPIPE, SIG_IGN); }

  // Kill any child the test left behind so no zombies outlive the suite.
  void TearDown() override { worker.terminate(); }

  WorkerProcess worker;
};

}  // namespace

// Spawn sets pid/running/fds, leaves isReady false (only the READY sentinel
// flips it), and the child answers a request round trip.
TEST_F(EmbeddingWorkerProcessTest, SpawnAndEchoRoundTrip) {
  ASSERT_TRUE(worker.spawn(0, echoChildMain));

  EXPECT_GT(worker.pid.load(), 0);
  EXPECT_TRUE(worker.running.load());
  EXPECT_FALSE(worker.isReady.load());
  EXPECT_TRUE(worker.checkAlive());

  const std::string request = R"([{"input":"hi","task_id":1}])";
  ASSERT_TRUE(worker.sendRequest(request));
  const auto response = worker.receiveResponse();
  EXPECT_EQ(std::string(response.begin(), response.end()), request);
}

// A child that exits immediately is detected by checkAlive, which reaps it
// and clears isReady.
TEST_F(EmbeddingWorkerProcessTest, CheckAliveDetectsCleanExit) {
  ASSERT_TRUE(worker.spawn(0, [](int, int) { _exit(0); }));
  worker.isReady.store(true);

  EXPECT_TRUE(waitUntilDead(worker));
  EXPECT_FALSE(worker.isReady.load());
}

// A nonzero exit code is detected the same way (the production log line
// differs; the liveness verdict must not).
TEST_F(EmbeddingWorkerProcessTest, CheckAliveDetectsNonzeroExit) {
  ASSERT_TRUE(worker.spawn(0, [](int, int) { _exit(3); }));

  EXPECT_TRUE(waitUntilDead(worker));
}

// A child killed by a signal is detected too.
TEST_F(EmbeddingWorkerProcessTest, CheckAliveDetectsSignalDeath) {
  ASSERT_TRUE(worker.spawn(0, echoChildMain));
  ASSERT_TRUE(worker.checkAlive());

  kill(worker.pid.load(), SIGKILL);

  EXPECT_TRUE(waitUntilDead(worker));
}

// terminate() kills a live child (no more signals deliverable to the pid)
// and closes both pipe ends.
TEST_F(EmbeddingWorkerProcessTest, TerminateKillsChildAndClosesPipes) {
  ASSERT_TRUE(worker.spawn(0, echoChildMain));
  const pid_t child = worker.pid.load();

  worker.terminate();

  EXPECT_NE(kill(child, 0), 0);  // reaped: probing the pid fails
  EXPECT_EQ(worker.writeFd.get(), -1);
  EXPECT_EQ(worker.readFd.get(), -1);
}

// terminate() on a never-spawned worker is a safe no-op; stop() calls it
// unconditionally.
TEST_F(EmbeddingWorkerProcessTest, TerminateWithoutSpawnIsNoOp) {
  worker.terminate();

  EXPECT_EQ(worker.pid.load(), -1);
}

// After the child dies, sendRequest fails and clears isReady instead of
// crashing the parent — what dispatchBatchToWorker relies on to fail a batch.
TEST_F(EmbeddingWorkerProcessTest, SendRequestFailsAfterChildDeath) {
  ASSERT_TRUE(worker.spawn(0, [](int, int) { _exit(0); }));
  ASSERT_TRUE(waitUntilDead(worker));
  worker.isReady.store(true);

  EXPECT_FALSE(worker.sendRequest("payload"));
  EXPECT_FALSE(worker.isReady.load());
}

// receiveResponse on a dead child returns empty and clears isReady.
TEST_F(EmbeddingWorkerProcessTest, ReceiveResponseFailsAfterChildDeath) {
  ASSERT_TRUE(worker.spawn(0, [](int, int) { _exit(0); }));
  ASSERT_TRUE(waitUntilDead(worker));
  worker.isReady.store(true);

  EXPECT_TRUE(worker.receiveResponse().empty());
  EXPECT_FALSE(worker.isReady.load());
}

// Closing the parent's request-pipe write end is the graceful shutdown
// signal: the child sees EOF and exits on its own, no SIGTERM needed.
TEST_F(EmbeddingWorkerProcessTest, ClosingRequestPipeShutsChildDown) {
  ASSERT_TRUE(worker.spawn(0, echoChildMain));
  ASSERT_TRUE(worker.checkAlive());

  worker.writeFd.reset();

  EXPECT_TRUE(waitUntilDead(worker));
}

// Two workers spawned side by side stay independent: each echoes its own
// message and terminating one leaves the other alive.
TEST_F(EmbeddingWorkerProcessTest, TwoWorkersAreIndependent) {
  WorkerProcess other;
  ASSERT_TRUE(worker.spawn(0, echoChildMain));
  ASSERT_TRUE(other.spawn(1, echoChildMain));

  ASSERT_TRUE(worker.sendRequest("first"));
  ASSERT_TRUE(other.sendRequest("second"));
  auto fromFirst = worker.receiveResponse();
  auto fromSecond = other.receiveResponse();
  EXPECT_EQ(std::string(fromFirst.begin(), fromFirst.end()), "first");
  EXPECT_EQ(std::string(fromSecond.begin(), fromSecond.end()), "second");

  other.terminate();
  EXPECT_TRUE(worker.checkAlive());
}
