// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "ipc/media_payload_ipc.hpp"
#include "runtime/runners/ipc_runner.hpp"

namespace tt::runners {

/** Generic child-side IPC loop for synchronous media runners. */
class MediaIpcRunner : public IRunner {
 public:
  MediaIpcRunner(std::string runnerName, int workerId);
  ~MediaIpcRunner() override;

  void stop() override;
  const char* runnerType() const override { return runner_name_.c_str(); }

 protected:
  int workerId() const { return worker_id_; }
  virtual void processTask(
      const tt::ipc::media_payload::MediaPayloadTask& task,
      tt::ipc::media_payload::MediaPayloadResult& result) = 0;

 private:
  void run() override;

  std::string runner_name_;
  int worker_id_;
  std::unique_ptr<tt::ipc::media_payload::MediaPayloadTaskQueue> task_queue_;
  std::unique_ptr<tt::ipc::media_payload::MediaPayloadResultQueue> result_queue_;
  std::atomic<bool> stopped_{false};
};

}  // namespace tt::runners
