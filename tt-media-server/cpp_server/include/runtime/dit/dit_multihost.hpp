// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>

#include "runtime/dit/dit_runner.hpp"

/**
 * One mesh across several hosts (for example a 4x32 quad), launched with
 * tt-run. Rank 0 serves HTTP; its worker wraps the runner in LockstepRunner,
 * which sends each request to the other ranks before running it locally. Ranks
 * 1..N-1 run Follower instead of the server: they open their part of the mesh,
 * warm up, and repeat every request rank 0 sends, discarding the output.
 *
 * The messages travel over tt-metal's MPI context (see MpiComm in
 * dit_python_runner.cpp): a request is its JSON, a shutdown is a marker.
 */
namespace tt::dit {

/** Blocking point-to-point messages from rank 0 to every other rank. */
class Comm {
 public:
  virtual ~Comm() = default;
  /** Rank 0 only. */
  virtual void broadcast(const std::string& message) = 0;
  /** Ranks 1..N-1: blocks until rank 0 broadcasts. */
  virtual std::string receive() = 0;
  virtual void barrier() = 0;
};

/** Messages are one type byte and a payload. */
std::string encodeRun(const DiTRequest& request);
std::string encodeShutdown();
bool isShutdown(const std::string& message);
/** Throws std::invalid_argument for anything but a run message. */
DiTRequest decodeRun(const std::string& message);

/** Exit the process: the ranks cannot be re-synchronised once one has
 *  diverged, and tt-run stops every rank when one exits. */
[[noreturn]] void endJob(const std::string& reason);

/** Rank 0's runner. The request is validated (and its pipeline call built)
 *  on rank 0 before it is sent, so a client's mistake of any kind reaches the
 *  client as an error and no other rank hears of it. After the broadcast a
 *  RequestError and an ExportError (only rank 0 exports) also reach the client;
 *  a failed broadcast or any other error leaves the ranks out of step and ends
 *  the job. */
class LockstepRunner : public IDiTRunner {
 public:
  using FatalHandler = std::function<void(const std::string&)>;
  LockstepRunner(
      std::unique_ptr<IDiTRunner> inner, std::shared_ptr<Comm> comm,
      FatalHandler onFatal = [](const std::string& r) { endJob(r); });

  void openDevice() override { inner_->openDevice(); }
  void warmup() override;
  void validate(const DiTRequest& request) override {
    inner_->validate(request);
  }
  DiTResult generate(const DiTRequest& request) override;
  void infer(const DiTRequest& request) override { inner_->infer(request); }
  void close() override;

 private:
  std::unique_ptr<IDiTRunner> inner_;
  std::shared_ptr<Comm> comm_;
  FatalHandler onFatal_;
  std::mutex lock_;
  bool followersReady_ = false;
};

/** Ranks 1..N-1. run() returns 0 after rank 0's shutdown, 1 on a fatal error
 *  (so tt-run stops the whole job instead of leaving rank 0 waiting in a
 *  collective this rank has abandoned). A RequestError is logged and the loop
 *  goes on; rank 0 reports it. */
class Follower {
 public:
  Follower(int rank, std::unique_ptr<IDiTRunner> runner,
           std::shared_ptr<Comm> comm)
      : rank_(rank), runner_(std::move(runner)), comm_(std::move(comm)) {}
  int run();

 private:
  int rank_;
  std::unique_ptr<IDiTRunner> runner_;
  std::shared_ptr<Comm> comm_;
};

}  // namespace tt::dit
