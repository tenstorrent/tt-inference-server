// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/video_service.hpp"

#include <json/json.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "metrics/metrics.hpp"
#include "utils/logger.hpp"

namespace tt::services {

namespace {

void writeJsonFile(const std::filesystem::path& path, const Json::Value& json) {
  Json::StreamWriterBuilder builder;
  builder["indentation"] = "";
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to open media IPC payload file: " +
                             path.string());
  }
  output << Json::writeString(builder, json);
}

Json::Value readJsonFile(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open media IPC response file: " +
                             path.string());
  }
  Json::CharReaderBuilder builder;
  Json::Value json;
  std::string errors;
  if (!Json::parseFromStream(builder, input, &json, &errors)) {
    throw std::runtime_error("failed to parse media IPC response file: " +
                             errors);
  }
  return json;
}

domain::video::VideoResponse responseFromJson(uint32_t taskId,
                                              const Json::Value& json) {
  domain::video::VideoResponse response(taskId);
  if (json.isMember("video_path") && json["video_path"].isString()) {
    response.videoPath = json["video_path"].asString();
  }
  if (json.isMember("generation_time") && json["generation_time"].isNumeric()) {
    response.generationTimeSeconds = json["generation_time"].asDouble();
  }
  if (json.isMember("error") && json["error"].isString()) {
    response.error = json["error"].asString();
  }
  if (response.videoPath.empty() && response.error.empty()) {
    response.error = "video response missing video_path";
  }
  return response;
}

constexpr size_t kMaxJobs = 10000;
constexpr int64_t kJobRetentionSeconds = 86400;  // Python job_retention_seconds

int64_t envInt64(const char* name, int64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  return (end != value && *end == '\0') ? static_cast<int64_t>(parsed)
                                        : fallback;
}

}  // namespace

VideoService::VideoService(
    config::VideoConfig config,
    std::unique_ptr<tt::worker::WorkerManager> workerManager,
    std::unique_ptr<tt::ipc::media_payload::MediaPayloadQueueSet> queueManager)
    : BaseAsyncService("VideoService", std::move(workerManager),
                       std::move(queueManager)),
      videoConfig(std::move(config)),
      payloadDir(std::filesystem::temp_directory_path() /
                 ("tt-video-ipc-" + std::to_string(::getpid()))),
      jobs(static_cast<size_t>(std::max<int64_t>(
               1, envInt64("MAX_JOBS", static_cast<int64_t>(kMaxJobs)))),
           envInt64("JOB_RETENTION_SECONDS", kJobRetentionSeconds)) {
  std::filesystem::create_directories(payloadDir);
  startSweeper();
  if (videoConfig.runner_type == config::ModelRunnerType::VIDEO_DEVICE) {
    // Delete only inside the output dir, and drop mp4 files left by a previous
    // run (the job table starts empty, so nothing here has a record).
    // TT_VIDEO_PURGE_ON_START=0 turns the purge off, e.g. when another server
    // shares this directory.
    const auto outDir = utils::resolveVideoOutputDir();
    jobs.setOutputRoot(outDir);
    outputDir_ = outDir;
    if (const std::string err = utils::checkDirWritable(outDir); !err.empty()) {
      // Not fatal: the problem may be fixed while the server runs, and every
      // request is checked again (503 until then).
      TT_LOG_ERROR(
          "[VideoService] Video output directory is not usable: {}. Requests "
          "get 503 until it is fixed.",
          err);
    }
    const char* purge = std::getenv("TT_VIDEO_PURGE_ON_START");
    if (purge == nullptr || std::string(purge) != "0") {
      const size_t removed = utils::purgeOrphanMp4(outDir);
      TT_LOG_INFO("[VideoService] Startup purge of '{}': removed {} orphan mp4",
                  outDir.string(), removed);
    }
  }
  TT_LOG_INFO(
      "[VideoService] Initialized worker-backed video service "
      "(workers={}, max_queue_size={}, payload_dir='{}')",
      numWorkers(), this->maxQueueSize, payloadDir.string());
}

VideoService::~VideoService() {
  stopSweeper();
  stop();
}

void VideoService::startSweeper() {
  sweepIntervalSeconds_ = envInt64("JOB_CLEANUP_INTERVAL_SECONDS", 300);
  maxStuckSeconds_ = envInt64("JOB_MAX_STUCK_TIME_SECONDS", 10800);
  if (sweepIntervalSeconds_ <= 0) return;
  sweeper_ = std::thread([this] {
    std::unique_lock<std::mutex> lock(sweeperMutex_);
    while (!sweeperCv_.wait_for(lock,
                                std::chrono::seconds(sweepIntervalSeconds_),
                                [this] { return sweeperStop_; })) {
      lock.unlock();
      const auto result = jobs.sweep(maxStuckSeconds_);
      tt::metrics::ServerMetrics::instance().onVideoSweep(result.evicted,
                                                          result.stuck);
      refreshJobGauges();
      if (result.evicted != 0 || result.stuck != 0) {
        TT_LOG_INFO("[VideoService] Job cleanup: evicted={} stuck_failed={}",
                    result.evicted, result.stuck);
      }
      lock.lock();
    }
  });
}

void VideoService::stopSweeper() {
  {
    std::lock_guard<std::mutex> lock(sweeperMutex_);
    sweeperStop_ = true;
  }
  sweeperCv_.notify_all();
  if (sweeper_.joinable()) sweeper_.join();
}

std::string VideoService::runnerInUse() const {
  if (!videoConfig.pythonRunner.empty()) return videoConfig.pythonRunner;
  return config::toClientRunnerName(videoConfig.runner_type);
}

// Python Job.model: SERVED_MODEL_NAME, else the model weights path.
std::string VideoService::servedModelName() const {
  if (const char* name = std::getenv("SERVED_MODEL_NAME");
      name != nullptr && *name != '\0') {
    return name;
  }
  if (!videoConfig.model_weights_path.empty()) {
    return videoConfig.model_weights_path;
  }
  return runnerInUse();
}

Json::Value VideoService::createJob(domain::VideoGenerateRequest request) {
  if (!isModelReady()) {
    throw std::runtime_error("Video service not ready");
  }
  const uint32_t taskId = request.task_id;
  // Python order (_enforceAdmissionLimits): outstanding jobs vs queue size
  // (429) first, then stored records vs max jobs (503). Both are checked in
  // create(), under the store lock, so two requests cannot both pass the check.
  Json::Value metadata;
  try {
    metadata = jobs.create(taskId, servedModelName(), request.toJson(), maxQueueSize);
  } catch (const ActiveJobLimitException&) {
    throw QueueFullException{};
  }
  {
    std::lock_guard<std::mutex> lock(submitMutex_);
    jobTasks_.insert(taskId);
  }
  try {
    submitRequestAsync(std::move(request),
                       [this, taskId](domain::video::VideoResponse&& r) {
                         // Counted by what the job ended as: a cancelled or
                         // stale job's discarded result is not an "ok" task.
                         if (const auto ok =
                                 jobs.complete(taskId, r.videoPath, r.error)) {
                           tt::metrics::ServerMetrics::instance()
                               .onVideoTaskDone(*ok, r.generationTimeSeconds,
                                                r.e2eSeconds);
                         }
                       });
  } catch (...) {
    {
      std::lock_guard<std::mutex> lock(submitMutex_);
      jobTasks_.erase(taskId);
    }
    jobs.erase(taskId);
    throw;
  }
  refreshJobGauges();
  return metadata;
}

std::string VideoService::outputDirError() const {
  if (outputDir_.empty()) return {};
  return utils::checkDirWritable(outputDir_);
}

std::optional<uint32_t> VideoService::resolveJobId(
    const std::string& publicId) const {
  return jobs.resolve(publicId);
}

std::optional<Json::Value> VideoService::getJob(uint32_t jobId) const {
  return jobs.get(jobId);
}

Json::Value VideoService::listJobs() const { return jobs.list(); }

std::optional<std::string> VideoService::jobResultPath(uint32_t jobId) const {
  return jobs.resultPath(jobId);
}

std::optional<Json::Value> VideoService::cancelJob(uint32_t jobId) {
  auto cancelled = jobs.cancel(jobId);
  if (cancelled) {
    // Tell the worker to skip the task if it has not started it yet. A task
    // that is already running is not interrupted (the result is dropped later).
    std::ofstream marker(tt::ipc::media_payload::cancelMarkerPath(
        payloadPath(jobId, "request"), jobId));
    if (!marker) {
      TT_LOG_WARN("[VideoService] Could not write cancel marker for task {}",
                  jobId);
    }
    // "cancelled": the job was still queued, the worker will skip its task.
    // "cancelling": the worker already runs it; the run is not interrupted and
    // its result is dropped when it returns.
    const std::string newStatus = (*cancelled)["status"].asString();
    TT_LOG_INFO("[VideoService] Cancel requested for task {}: job is now {}{}",
                jobId, newStatus,
                newStatus == "cancelling"
                    ? " (already running, result will be discarded)"
                    : " (queued, worker will skip it)");
  }
  refreshJobGauges();
  return cancelled;
}

// The cancel marker is deliberately left in place: a job cancelled while queued
// is terminal (so it can be deleted) but its task still sits in the shm queue,
// and the worker must still see the marker to skip it. cleanupTask() removes it
// when the worker's result for that task arrives.
bool VideoService::deleteJob(uint32_t jobId) {
  const bool removed = jobs.remove(jobId);
  refreshJobGauges();
  return removed;
}

void VideoService::removeCancelMarker(uint32_t taskId) const {
  std::error_code ec;
  std::filesystem::remove(tt::ipc::media_payload::cancelMarkerPath(
                              payloadPath(taskId, "request"), taskId),
                          ec);
}

void VideoService::onTaskClaimed(
    size_t /*workerIdx*/, const IpcResult& result,
    std::chrono::steady_clock::time_point /*claimTime*/, double queueWaitMs,
    double shmEnqueueMs) {
  auto& metrics = tt::metrics::ServerMetrics::instance();
  metrics.onVideoTaskClaimed(queueWaitMs / 1000.0);
  metrics.onVideoOverhead(
      tt::metrics::ServerMetrics::VideoOverheadPhase::ShmEnqueue,
      shmEnqueueMs / 1000.0);
  jobs.markInProgress(result.getTaskId());
  refreshJobGauges();
}

void VideoService::onTaskCompleted(
    size_t /*workerIdx*/, const IpcResult& /*result*/,
    std::chrono::steady_clock::time_point gotResultTime,
    std::chrono::steady_clock::time_point doneTime) {
  tt::metrics::ServerMetrics::instance().onVideoOverhead(
      tt::metrics::ServerMetrics::VideoOverheadPhase::ResponseHandling,
      std::chrono::duration<double>(doneTime - gotResultTime).count());
  refreshJobGauges();
}

void VideoService::refreshJobGauges() const {
  tt::metrics::ServerMetrics::instance().setVideoJobs(jobs.countByStatus());
}

std::string VideoService::validateRequest(
    const domain::VideoGenerateRequest& /*request*/) const {
  if (!isModelReady()) {
    return "Video service not ready";
  }
  return {};
}

VideoService::IpcTask VideoService::buildIpcTask(
    const domain::VideoGenerateRequest& request) {
  const auto submitTime = std::chrono::steady_clock::now();
  const auto reqPath = payloadPath(request.task_id, "request");
  const auto respPath = payloadPath(request.task_id, "response");
  writeJsonFile(reqPath, request.toJson());
  tt::metrics::ServerMetrics::instance().onVideoOverhead(
      tt::metrics::ServerMetrics::VideoOverheadPhase::RequestWrite,
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    submitTime)
          .count());
  {
    std::lock_guard<std::mutex> lock(submitMutex_);
    submitTimes_[request.task_id] = submitTime;
  }

  IpcTask task;
  task.task_id = request.task_id;
  task.request_path = reqPath;
  task.response_path = respPath;
  return task;
}

domain::video::VideoResponse VideoService::buildResponse(
    uint32_t taskId, const IpcResult& result) {
  domain::video::VideoResponse response(taskId);
  response.generationTimeSeconds = result.generation_time_seconds;
  if (!result.error.empty()) {
    response.error = result.error;
  } else {
    try {
      auto body = readJsonFile(payloadPath(taskId, "response"));
      response = responseFromJson(taskId, body);
      response.generationTimeSeconds = result.generation_time_seconds;
    } catch (const std::exception& e) {
      response.error = e.what();
    }
  }
  // Telemetry: submit -> result popped (called before cleanupTask).
  double e2eSeconds = 0.0;
  {
    std::lock_guard<std::mutex> lock(submitMutex_);
    auto it = submitTimes_.find(taskId);
    if (it != submitTimes_.end()) {
      e2eSeconds = std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - it->second)
                       .count();
    }
  }
  response.e2eSeconds = e2eSeconds;
  // A job's outcome is only known once jobs.complete() has applied the final
  // status (a cancelled or stale job's result is discarded), so a job records
  // its outcome in createJob(). A request without a job (sync mode) is done now.
  bool belongsToJob = false;
  {
    std::lock_guard<std::mutex> lock(submitMutex_);
    belongsToJob = jobTasks_.count(taskId) > 0;
  }
  if (!belongsToJob) {
    tt::metrics::ServerMetrics::instance().onVideoTaskDone(
        response.error.empty(), result.generation_time_seconds, e2eSeconds);
  }
  return response;
}

void VideoService::cleanupTask(uint32_t taskId) {
  {
    std::lock_guard<std::mutex> lock(submitMutex_);
    submitTimes_.erase(taskId);
    jobTasks_.erase(taskId);
  }
  std::error_code ec;
  std::filesystem::remove(payloadPath(taskId, "request"), ec);
  std::filesystem::remove(payloadPath(taskId, "response"), ec);
  removeCancelMarker(taskId);
}

void VideoService::onStopped() {
  std::error_code ec;
  std::filesystem::remove_all(payloadDir, ec);
}

std::string VideoService::payloadPath(uint32_t taskId,
                                      const char* prefix) const {
  return (payloadDir /
          (std::string(prefix) + "-" + std::to_string(taskId) + ".json"))
      .string();
}

}  // namespace tt::services
