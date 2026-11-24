// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "config/runner_config.hpp"
#include "domain/video/video_response.hpp"
#include "domain/video_generate_request.hpp"
#include "ipc/media_payload_ipc.hpp"
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
 * MODEL_RUNNER, and the JSON `video_path` is the mp4 written for that job. */
class VideoService
    : public BaseAsyncService<domain::VideoGenerateRequest,
                              domain::video::VideoResponse,
                              tt::ipc::media_payload::MediaPayloadQueueSet> {
 public:
  VideoService(config::VideoConfig config,
               std::unique_ptr<tt::worker::WorkerManager> workerManager,
               std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet>
                   queueManager);
  ~VideoService() override;

  std::string runnerInUse() const override;
  std::string servedModelName() const;

  /** Enqueue and return the job metadata (status "queued"). Throws
   *  QueueFullException, MaxJobsReachedException, or std::runtime_error when
   *  the model is not ready. */
  Json::Value createJob(domain::VideoGenerateRequest request);

  /** Public job id (UUID) -> internal task id; nullopt = unknown/malformed. */
  std::optional<uint32_t> resolveJobId(const std::string& publicId) const;
  std::optional<Json::Value> getJob(uint32_t jobId) const;
  Json::Value listJobs() const;
  std::optional<std::string> jobResultPath(uint32_t jobId) const;
  /** Whether a result file the runner reported may be served: the same rule a
   *  job's result goes through (inside the output directory). Used by the sync
   *  path, which has no job. */
  bool isResultPathAllowed(const std::string& path) const {
    return jobs.isResultPathAllowed(path);
  }
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
  void onStopped() override;

 private:
  std::string payloadPath(uint32_t taskId, const char* prefix) const;
  void removeCancelMarker(uint32_t taskId) const;

  config::VideoConfig videoConfig;
  std::filesystem::path payloadDir;
  void startSweeper();
  void stopSweeper();

  VideoJobStore jobs;

  // Periodic job cleanup (Python job_cleanup_interval_seconds etc.).
  int64_t sweepIntervalSeconds_ = 300;
  int64_t maxStuckSeconds_ = 10800;
  std::thread sweeper_;
  std::mutex sweeperMutex_;
  std::condition_variable sweeperCv_;
  bool sweeperStop_ = false;
};

}  // namespace tt::services
