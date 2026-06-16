// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <atomic>
#include <cstdint>
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
  const char* runnerType() const override { return runnerName.c_str(); }

 protected:
  int workerId() const { return workerIndex; }
  virtual Json::Value processJsonTask(const Json::Value& request,
                                      uint32_t taskId) = 0;

 private:
  void run() override;
  void processTask(const tt::ipc::media_payload::MediaPayloadTask& task,
                   tt::ipc::media_payload::MediaPayloadResult& result);

  std::string runnerName;
  int workerIndex;
  std::unique_ptr<tt::ipc::media_payload::MediaPayloadTaskQueue> taskQueue;
  std::unique_ptr<tt::ipc::media_payload::MediaPayloadResultQueue> resultQueue;
  std::atomic<bool> stopped{false};
};

}  // namespace tt::runners
