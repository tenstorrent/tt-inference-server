// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>

#include "config/runner_config.hpp"
#include "domain/video/video_response.hpp"
#include "domain/video_generate_request.hpp"
#include "ipc/media_payload_ipc.hpp"
#include "runtime/runners/media_runner.hpp"
#include "runtime/worker/worker_manager.hpp"
#include "services/request_pipeline.hpp"
#include "services/video_job_store.hpp"

namespace tt::services {

/** Video service backed by WorkerManager + MediaPayloadQueueSet.
 *
 * Job based, like the Python server: `createJob` enqueues the request and
 * returns the job metadata (id, status=queued) immediately; the client polls
 * `getJob` until `completed` and then downloads the mp4 via `resultPath`.
 * The device runner opens the chip, runs the Python DiT model selected by
 * MODEL_RUNNER, and the JSON `video_path` is the mp4 written for that job.
 *
 * Two ways to run the device runner:
 *  - worker process(es) (several DEVICE_IDS groups, or a multihost launch):
 *    tasks travel through the shared-memory queue as described above;
 *  - in process (one DEVICE_IDS group): the runner lives on a thread of this
 *    server process and the queue is a plain deque in memory. No shared
 *    memory, no request/response files, no second process. A cancelled job that
 *    is still queued is removed from the deque. Chosen by `useInProcess()`. */
class VideoService
    : public BaseAsyncService<domain::VideoGenerateRequest,
                              domain::video::VideoResponse,
                              tt::ipc::media_payload::MediaPayloadQueueSet> {
 public:
  VideoService(config::VideoConfig config,
               std::unique_ptr<tt::worker::WorkerManager> workerManager,
               std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet>
                   queueManager);
  /** In-process mode: no WorkerManager, no shared-memory queues. The runner is
   *  created and warmed up on the service's own thread (start()). */
  explicit VideoService(config::VideoConfig config);
  ~VideoService() override;

  /** True when the service should run in process: TT_VIDEO_IN_PROCESS=1, or
   *  not set and there is a single worker (one DEVICE_IDS group) on a launch
   *  that is not multihost. TT_VIDEO_IN_PROCESS=0 keeps the worker process.
   *  Throws std::runtime_error for =1 with several workers or on multihost. */
  static bool useInProcess();

  void start() override;
  void stop() override;
  bool isModelReady() const override;
  bool hasFatalError() const override;

  /** Same as the base class, plus the in-process queue. Used by the sync
   *  (`?async=false`) endpoint. */
  void submitRequestAsync(domain::VideoGenerateRequest request,
                          Callback onComplete);

  std::string runnerInUse() const override;
  std::string servedModelName() const;

  /** Enqueue and return the job metadata (status "queued"). Throws
   *  QueueFullException, MaxJobsReachedException, or std::runtime_error when
   *  the model is not ready. */
  Json::Value createJob(domain::VideoGenerateRequest request);

  /** Empty when new work can be accepted; otherwise why the mp4 output
   *  directory cannot be written (the controller answers 503). Only checked
   *  for the real device runner; the mock runner writes no files. */
  std::string outputDirError() const;

  /** Public job id (UUID) -> internal task id; nullopt = unknown/malformed. */
  std::optional<uint32_t> resolveJobId(const std::string& publicId) const;
  std::optional<Json::Value> getJob(uint32_t jobId) const;
  Json::Value listJobs() const;
  std::optional<std::string> jobResultPath(uint32_t jobId) const;
  /** nullopt = unknown or already terminal. */
  std::optional<Json::Value> cancelJob(uint32_t jobId);
  /** false = unknown; throws JobStillActiveException if not terminal. */
  bool deleteJob(uint32_t jobId);

 protected:
  std::string validateRequest(
      const domain::VideoGenerateRequest& request) const override;
  IpcTask buildIpcTask(const domain::VideoGenerateRequest& request) override;
  domain::video::VideoResponse buildResponse(uint32_t taskId,
                                             const IpcResult& result) override;
  void cleanupTask(uint32_t taskId) override;
  void onTaskClaimed(size_t workerIdx, const IpcResult& result,
                     std::chrono::steady_clock::time_point claimTime,
                     double queueWaitMs, double shmEnqueueMs) override;
  void onTaskCompleted(size_t workerIdx, const IpcResult& result,
                       std::chrono::steady_clock::time_point gotResultTime,
                       std::chrono::steady_clock::time_point doneTime) override;
  void onStopped() override;
  bool tryRecoverWorker(size_t workerIdx) override;
  size_t currentQueueSize() const override;
  std::vector<tt::worker::WorkerInfo> getWorkerInfo() const override;

 private:
  std::string payloadPath(uint32_t taskId, const char* prefix) const;
  void removeCancelMarker(uint32_t taskId) const;

  config::VideoConfig videoConfig;
  std::filesystem::path payloadDir;
  std::filesystem::path outputDir_;  // empty when the runner writes no mp4
  void startSweeper();
  void stopSweeper();

  VideoJobStore jobs;

  // Telemetry: when each task was handed to the pipeline (for e2e time).
  void refreshJobGauges() const;
  std::mutex submitMutex_;
  std::unordered_map<uint32_t, std::chrono::steady_clock::time_point>
      submitTimes_;

  // Periodic job cleanup (Python job_cleanup_interval_seconds etc.).
  int64_t sweepIntervalSeconds_ = 300;
  int64_t maxStuckSeconds_ = 10800;
  std::thread sweeper_;
  std::mutex sweeperMutex_;
  std::condition_variable sweeperCv_;
  bool sweeperStop_ = false;

  // ---- in-process mode (inProcess_) ----------------------------------------
  struct InProcessTask {
    domain::VideoGenerateRequest request;
    Callback onComplete;
    std::chrono::steady_clock::time_point submitTime;
  };
  void enqueueInProcess(domain::VideoGenerateRequest request,
                        Callback onComplete);
  void inProcessLoop();
  void serveInProcessTask(InProcessTask& task);
  void failQueuedInProcess(const std::string& error);
  domain::video::VideoResponse runInProcessTask(const InProcessTask& task);

  bool inProcess_ = false;
  std::unique_ptr<
      tt::runners::IMediaRunner<domain::VideoGenerateRequest, std::string>>
      inRunner_;
  std::thread inThread_;
  mutable std::mutex inMutex_;  // guards inQueue_, inRunning_, inStop_
  std::condition_variable inCv_;
  std::deque<InProcessTask> inQueue_;
  std::optional<uint32_t> inRunning_;  // task the runner is working on
  bool inStop_ = false;
  bool inStarted_ = false;
  std::atomic<bool> inReady_{false};
  std::atomic<bool> inFailed_{false};
  std::atomic<bool> inLoopExited_{false};
  mutable std::mutex inFailureMutex_;
  std::string inFailureReason_;

  // Automatic worker restart (not used for multihost launches).
  size_t maxWorkerRestarts_ = 3;
  std::mutex restartMutex_;
  std::deque<std::chrono::steady_clock::time_point> restartTimes_;
};

}  // namespace tt::services
