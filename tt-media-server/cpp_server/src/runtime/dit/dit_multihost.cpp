// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/dit/dit_multihost.hpp"

#include <unistd.h>

#include <cstdio>
#include <cstdlib>

#include "utils/logger.hpp"

namespace tt::dit {

namespace {
constexpr char kRunTag = 'R';
constexpr char kShutdownTag = 'S';
}  // namespace

std::string encodeRun(const DiTRequest& request) {
  Json::StreamWriterBuilder writer;
  writer["indentation"] = "";
  std::string message(1, kRunTag);
  message += Json::writeString(writer, request.toJson());
  return message;
}

std::string encodeShutdown() { return std::string(1, kShutdownTag); }

bool isShutdown(const std::string& message) {
  return !message.empty() && message[0] == kShutdownTag;
}

DiTRequest decodeRun(const std::string& message) {
  if (message.empty() || message[0] != kRunTag) {
    throw std::invalid_argument("unknown message from rank 0");
  }
  Json::Value json;
  Json::CharReaderBuilder builder;
  std::string errors;
  std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
  if (!reader->parse(message.data() + 1, message.data() + message.size(), &json,
                     &errors)) {
    throw std::invalid_argument("bad run message: " + errors);
  }
  return DiTRequest::fromJson(json);
}

void endJob(const std::string& reason) {
  TT_LOG_ERROR("[DiTMultihost] Rank 0: multi-host job cannot continue: {}",
               reason);
  std::fflush(nullptr);
  ::_exit(1);
}

LockstepRunner::LockstepRunner(std::unique_ptr<IDiTRunner> inner,
                               std::shared_ptr<Comm> comm, FatalHandler onFatal)
    : inner_(std::move(inner)),
      comm_(std::move(comm)),
      onFatal_(std::move(onFatal)) {}

void LockstepRunner::warmup() {
  inner_->warmup();
  // Followers warm up on their own and then wait here, so the worker reports
  // ready only once every rank is warm.
  TT_LOG_INFO("[DiTMultihost] Rank 0: warm, waiting for the other ranks");
  comm_->barrier();
  followersReady_ = true;
  TT_LOG_INFO("[DiTMultihost] Rank 0: all ranks warm");
}

DiTResult LockstepRunner::generate(const DiTRequest& request) {
  std::lock_guard<std::mutex> guard(lock_);
  inner_->validate(request);  // nothing is sent for a request rank 0 refuses
  try {
    comm_->broadcast(encodeRun(request));
  } catch (const std::exception& e) {
    onFatal_(std::string("broadcast to the other ranks failed: ") + e.what());
    throw;
  }
  try {
    return inner_->generate(request);
  } catch (const RequestError&) {
    throw;
  } catch (const ExportError&) {
    throw;  // the pipeline ran on every rank; only rank 0's export failed
  } catch (const std::exception& e) {
    onFatal_(std::string("run failed: ") + e.what());
    throw;
  }
}

void LockstepRunner::close() {
  std::unique_lock<std::mutex> guard(lock_, std::defer_lock);
  // Wait a little for an in-flight request, then close anyway.
  for (int i = 0; i < 50 && !guard.try_lock(); ++i) ::usleep(100'000);
  // Before the warmup barrier the followers are not listening yet; tt-run
  // tears them down when this rank exits.
  if (followersReady_) {
    try {
      comm_->broadcast(encodeShutdown());
    } catch (const std::exception& e) {
      TT_LOG_WARN("[DiTMultihost] Rank 0: failed to send shutdown: {}",
                  e.what());
    }
  }
  inner_->close();
}

int Follower::run() {
  try {
    TT_LOG_INFO("[DiTMultihost] Rank {}: opening mesh", rank_);
    runner_->openDevice();
    TT_LOG_INFO("[DiTMultihost] Rank {}: warming up", rank_);
    runner_->warmup();
    TT_LOG_INFO("[DiTMultihost] Rank {}: warm, waiting for all ranks", rank_);
    comm_->barrier();
    TT_LOG_INFO("[DiTMultihost] Rank {}: following rank 0", rank_);
    for (;;) {
      const std::string message = comm_->receive();
      if (isShutdown(message)) break;
      const DiTRequest request = decodeRun(message);
      try {
        runner_->infer(request);
      } catch (const RequestError& e) {
        TT_LOG_ERROR("[DiTMultihost] Rank {}: task {} refused: {}", rank_,
                     request.taskId, e.what());
      }
    }
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[DiTMultihost] Rank {}: fatal error: {}", rank_, e.what());
    return 1;
  }
  TT_LOG_INFO("[DiTMultihost] Rank {}: shutdown from rank 0", rank_);
  try {
    runner_->close();
  } catch (const std::exception& e) {
    TT_LOG_WARN("[DiTMultihost] Rank {}: close failed: {}", rank_, e.what());
  }
  return 0;
}

}  // namespace tt::dit
