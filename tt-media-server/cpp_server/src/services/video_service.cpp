// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/video_service.hpp"

#include <dlfcn.h>
#include <json/json.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "config/defaults.hpp"
#include "config/settings.hpp"
#include "metrics/metrics.hpp"
#include "runtime/runners/runner_registry.hpp"
#include "runtime/runners/video_device_runner.hpp"
#include "runtime/runners/video_python_env.hpp"
#include "utils/logger.hpp"
#include "utils/video_faststart.hpp"

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
constexpr int64_t kRestartWindowSeconds = 3600;

int64_t envInt64(const char* name, int64_t fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') return fallback;
  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  return (end != value && *end == '\0') ? static_cast<int64_t>(parsed)
                                        : fallback;
}

// Same as VideoIpcRunner::processJsonTask: the finished mp4 is made faststart
// once, here, so /download can stream the file as it is.
void faststartFinishedMp4(const std::string& path, uint32_t taskId) {
  if (path.size() <= 4 || path.rfind("mock://", 0) == 0 ||
      path.compare(path.size() - 4, 4, ".mp4") != 0) {
    return;
  }
  switch (tt::utils::faststartInPlace(path)) {
    case tt::utils::FaststartResult::AlreadyFaststart:
      TT_LOG_INFO("[VideoService] task {} mp4 already faststart", taskId);
      break;
    case tt::utils::FaststartResult::Remuxed:
      TT_LOG_INFO("[VideoService] task {} faststart remux done", taskId);
      break;
    case tt::utils::FaststartResult::Failed:
      TT_LOG_WARN(
          "[VideoService] task {} faststart remux failed, keeping original",
          taskId);
      break;
  }
}

// The environment a worker process gets from WorkerManager
// (populateCommonWorkerEnv for video), set in this process before the runner
// starts when the runner lives on a thread of the server.
void setWorkerEnvInProcess(const config::VideoConfig& cfg) {
  ::setenv("TT_WORKER_ID", std::to_string(cfg.worker_id).c_str(), 1);
  ::setenv("TT_VISIBLE_DEVICES", cfg.visible_devices.c_str(), 1);
  ::setenv("OMP_NUM_THREADS", "2", 1);
  ::setenv("MKL_NUM_THREADS", "2", 1);
  ::setenv("TORCH_NUM_THREADS", "1", 1);
  if (const char* throttle = std::getenv("DEFAULT_THROTTLE_LEVEL");
      throttle && *throttle) {
    ::setenv("TT_MM_THROTTLE_PERF", throttle, 1);
  }
  if (const char* metalHome = std::getenv("TT_METAL_HOME");
      metalHome && *metalHome) {
    std::string deviceSuffix = cfg.visible_devices;
    std::replace(deviceSuffix.begin(), deviceSuffix.end(), ',', '_');
    ::setenv("TT_METAL_CACHE",
             (std::string(metalHome) + "/built/" + deviceSuffix).c_str(), 1);
  }
}

// tt-metal locks every chip with a robust mutex owned by the thread that opened
// the device, and releases it when its MetalContext is destroyed. Left to the
// atexit handler that runs on the main thread, that unlock fails (EPERM) and
// the process aborts at exit. So the runner thread, which opened the device,
// destroys the contexts itself (tt::tt_metal::detail::ReleaseOwnership,
// tt_metal.hpp) once the runner has closed its mesh. libtt_metal is loaded by
// the embedded Python (ttnn); nothing to do when it is not loaded (the mock
// runner).
void releaseDeviceContextOnThisThread() {
  void* lib = ::dlopen("libtt_metal.so", RTLD_LAZY | RTLD_NOLOAD);
  if (lib == nullptr) return;
  using ReleaseFn = void (*)();
  auto release = reinterpret_cast<ReleaseFn>(
      ::dlsym(lib, "_ZN2tt8tt_metal6detail16ReleaseOwnershipEv"));
  if (release == nullptr) {
    TT_LOG_WARN(
        "[VideoService] tt::tt_metal::detail::ReleaseOwnership not found; the "
        "device context is left to the exit handler");
  } else {
    release();
    TT_LOG_INFO("[VideoService] Device context released on the runner thread");
  }
  ::dlclose(lib);
}

}  // namespace

bool VideoService::useInProcess() {
  const bool multihost = envInt64("OMPI_COMM_WORLD_SIZE", 1) > 1;
  const bool single = config::numWorkers() == 1;
  const char* value = std::getenv("TT_VIDEO_IN_PROCESS");
  if (value == nullptr || *value == '\0') return single && !multihost;
  const std::string mode = value;
  if (mode == "0" || mode == "false") return false;
  if (mode == "1" || mode == "true") {
    if (!single || multihost) {
      throw std::runtime_error(
          "[VideoService] TT_VIDEO_IN_PROCESS=1 needs exactly one DEVICE_IDS "
          "group and no multihost launch");
    }
    return true;
  }
  throw std::runtime_error("[VideoService] TT_VIDEO_IN_PROCESS must be 0 or 1");
}

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
  maxWorkerRestarts_ = static_cast<size_t>(
      std::max<int64_t>(0, envInt64("TT_VIDEO_WORKER_MAX_RESTARTS", 3)));
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

// In-process mode. The same limits, sweeper and output directory handling as
// the worker-backed constructor, but no payload directory (requests travel as
// objects, not files) and no WorkerManager.
VideoService::VideoService(config::VideoConfig config)
    : BaseAsyncService("VideoService", nullptr, nullptr),
      videoConfig(std::move(config)),
      jobs(static_cast<size_t>(std::max<int64_t>(
               1, envInt64("MAX_JOBS", static_cast<int64_t>(kMaxJobs)))),
           envInt64("JOB_RETENTION_SECONDS", kJobRetentionSeconds)),
      inProcess_(true) {
  videoConfig.worker_id = 0;
  videoConfig.visible_devices = config::visibleDevicesForWorker(0);
  startSweeper();
  if (videoConfig.runner_type == config::ModelRunnerType::VIDEO_DEVICE) {
    const auto outDir = utils::resolveVideoOutputDir();
    jobs.setOutputRoot(outDir);
    outputDir_ = outDir;
    if (const std::string err = utils::checkDirWritable(outDir); !err.empty()) {
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
      "[VideoService] Initialized in-process video service (runner on a "
      "thread of this process, in-memory queue, max_queue_size={}, "
      "devices='{}')",
      this->maxQueueSize, videoConfig.visible_devices);
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

// A dead video worker is restarted (the shm task queue keeps the queued tasks,
// the new worker takes them after its warmup) unless: it is a multihost launch
// (all ranks run in lockstep, one cannot be replaced alone), restarts are
// disabled (TT_VIDEO_WORKER_MAX_RESTARTS=0), or it already restarted that many
// times within the last hour (a crash loop). Then the service fails as before
// and the process exits so a supervisor can restart it.
bool VideoService::tryRecoverWorker(size_t workerIdx) {
  if (maxWorkerRestarts_ == 0 || !workerManager()) return false;
  if (envInt64("OMPI_COMM_WORLD_SIZE", 1) > 1) return false;
  {
    std::lock_guard<std::mutex> lock(restartMutex_);
    const auto now = std::chrono::steady_clock::now();
    while (!restartTimes_.empty() &&
           now - restartTimes_.front() >
               std::chrono::seconds(kRestartWindowSeconds)) {
      restartTimes_.pop_front();
    }
    if (restartTimes_.size() >= maxWorkerRestarts_) {
      TT_LOG_ERROR(
          "[VideoService] Worker {} died; {} restarts within the last hour, "
          "giving up",
          workerIdx, restartTimes_.size());
      tt::metrics::ServerMetrics::instance().onVideoWorkerRestart("gave_up");
      return false;
    }
    restartTimes_.push_back(now);
  }
  try {
    workerManager()->restartWorker(workerIdx);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[VideoService] Restart of worker {} failed: {}", workerIdx,
                 e.what());
    tt::metrics::ServerMetrics::instance().onVideoWorkerRestart("failed");
    return false;
  }
  tt::metrics::ServerMetrics::instance().onVideoWorkerRestart("ok");
  TT_LOG_WARN("[VideoService] Worker {} died and was restarted", workerIdx);
  return true;
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
  // (429) first, then stored records vs max jobs (503, thrown by create()).
  if (jobs.activeCount() >= maxQueueSize) throw QueueFullException{};
  auto metadata = jobs.create(taskId, servedModelName(),
                              request.toJson(/*withMedia=*/false));
  try {
    submitRequestAsync(std::move(request),
                       [this, taskId](domain::video::VideoResponse&& r) {
                         jobs.complete(taskId, r.videoPath, r.error);
                       });
  } catch (...) {
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
  if (inProcess_) {
    // The queue is in this process: a job that is still queued is taken out of
    // it, so no cancel marker is needed. The lock makes "queued in the job
    // table" and "still in the deque" one decision (the runner thread marks a
    // job in progress under the same lock when it takes it).
    std::optional<Json::Value> cancelled;
    bool removedFromQueue = false;
    {
      std::lock_guard<std::mutex> lock(inMutex_);
      cancelled = jobs.cancel(jobId);
      if (cancelled) {
        auto it = std::find_if(inQueue_.begin(), inQueue_.end(),
                               [jobId](const InProcessTask& task) {
                                 return task.request.task_id == jobId;
                               });
        if (it != inQueue_.end()) {
          inQueue_.erase(it);
          removedFromQueue = true;
        }
      }
    }
    if (cancelled) {
      TT_LOG_INFO(
          "[VideoService] Cancel requested for task {}: job is now {}{}", jobId,
          (*cancelled)["status"].asString(),
          removedFromQueue
              ? " (queued, removed from the queue)"
              : ((*cancelled)["status"].asString() == "cancelling"
                     ? " (already running, result will be discarded)"
                     : " (queued, not in the queue yet; the runner will skip "
                       "it)"));
    }
    refreshJobGauges();
    return cancelled;
  }
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
    size_t workerIdx, const IpcResult& result,
    std::chrono::steady_clock::time_point /*claimTime*/, double queueWaitMs,
    double shmEnqueueMs) {
  auto& metrics = tt::metrics::ServerMetrics::instance();
  metrics.onVideoTaskClaimed(queueWaitMs / 1000.0);
  metrics.onVideoOverhead(
      tt::metrics::ServerMetrics::VideoOverheadPhase::ShmEnqueue,
      shmEnqueueMs / 1000.0);
  // Remember which worker process holds the job (Python's
  // `worker_assignment.identity`), for stopping or replacing it later.
  int pid = 0;
  if (auto* manager = workerManager()) {
    if (auto* worker = manager->worker(workerIdx)) {
      pid = static_cast<int>(worker->pid.load());
    }
  }
  jobs.markInProgress(result.getTaskId(), static_cast<int>(workerIdx), pid);
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
  tt::metrics::ServerMetrics::instance().onVideoTaskDone(
      response.error.empty(), result.generation_time_seconds, e2eSeconds);
  return response;
}

void VideoService::cleanupTask(uint32_t taskId) {
  {
    std::lock_guard<std::mutex> lock(submitMutex_);
    submitTimes_.erase(taskId);
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

// ---------------------------------------------------------------------------
// In-process mode
// ---------------------------------------------------------------------------

void VideoService::start() {
  if (!inProcess_) {
    BaseAsyncService::start();
    return;
  }
  {
    std::lock_guard<std::mutex> lock(inMutex_);
    if (inStarted_) return;
    inStarted_ = true;
    inStop_ = false;
  }
  // The runner thread and the HTTP threads read the environment while they
  // run; setenv() is only safe before they exist, so it all happens here, on
  // the thread that starts the service.
  setWorkerEnvInProcess(videoConfig);
  if (videoConfig.runner_type == config::ModelRunnerType::VIDEO_DEVICE) {
    tt::runners::VideoDeviceRunner::prepareProcessEnvironment(videoConfig);
  }
  inThread_ = std::thread([this] { inProcessLoop(); });
}

void VideoService::stop() {
  if (!inProcess_) {
    BaseAsyncService::stop();
    return;
  }
  {
    std::lock_guard<std::mutex> lock(inMutex_);
    if (!inStarted_) return;
    inStop_ = true;
  }
  inCv_.notify_all();
  if (inThread_.joinable()) {
    TT_LOG_INFO("[VideoService] Stopping the in-process runner...");
    // A running generation cannot be interrupted. Give it the same grace a
    // worker process gets; after that exit, so the device is not held by a
    // process that is shutting down.
    namespace defs = tt::config::defaults;
    for (unsigned waited = 0; waited < defs::WORKER_SHUTDOWN_GRACE_MS &&
                              !inLoopExited_.load(std::memory_order_acquire);
         waited += defs::SHUTDOWN_POLL_MS) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(defs::SHUTDOWN_POLL_MS));
    }
    if (!inLoopExited_.load(std::memory_order_acquire)) {
      TT_LOG_CRITICAL(
          "[VideoService] The runner did not finish within {} ms of stop(); "
          "exiting",
          defs::WORKER_SHUTDOWN_GRACE_MS);
      // _Exit skips destructors, so main()'s WorkerMetricsShm would not
      // unlink its segment; do it here (ENOENT is fine).
      shm_unlink(tt::config::workerMetricsShmName().c_str());
      std::_Exit(1);
    }
    inThread_.join();
  }
  failQueuedInProcess("Server shutting down");
  inRunner_.reset();  // already stopped by the runner thread
  inReady_.store(false, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(inMutex_);
    inStarted_ = false;
  }
  TT_LOG_INFO("[VideoService] Stopped");
}

bool VideoService::isModelReady() const {
  if (!inProcess_) return BaseAsyncService::isModelReady();
  return inReady_.load(std::memory_order_acquire) &&
         !inFailed_.load(std::memory_order_acquire);
}

bool VideoService::hasFatalError() const {
  if (!inProcess_) return BaseAsyncService::hasFatalError();
  return inFailed_.load(std::memory_order_acquire);
}

size_t VideoService::currentQueueSize() const {
  if (!inProcess_) return BaseAsyncService::currentQueueSize();
  std::lock_guard<std::mutex> lock(inMutex_);
  return inQueue_.size() + (inRunning_ ? 1 : 0);
}

std::vector<tt::worker::WorkerInfo> VideoService::getWorkerInfo() const {
  if (!inProcess_) return BaseAsyncService::getWorkerInfo();
  tt::worker::WorkerInfo info;
  info.worker_id = "0";
  info.is_ready = isModelReady();
  info.is_alive = !inFailed_.load(std::memory_order_acquire);
  info.pid = ::getpid();
  return {info};
}

void VideoService::submitRequestAsync(domain::VideoGenerateRequest request,
                                      Callback onComplete) {
  if (!inProcess_) {
    BaseAsyncService::submitRequestAsync(std::move(request),
                                         std::move(onComplete));
    return;
  }
  this->preProcess(request);  // queue full -> QueueFullException
  domain::video::VideoResponse refused(request.task_id);
  if (inFailed_.load(std::memory_order_acquire)) {
    std::lock_guard<std::mutex> lock(inFailureMutex_);
    refused.error =
        inFailureReason_.empty() ? "Video runner failed" : inFailureReason_;
    onComplete(std::move(refused));
    return;
  }
  if (!isModelReady()) {
    refused.error = "Video service not ready";
    onComplete(std::move(refused));
    return;
  }
  enqueueInProcess(std::move(request), std::move(onComplete));
}

void VideoService::enqueueInProcess(domain::VideoGenerateRequest request,
                                    Callback onComplete) {
  bool stopping = false;
  {
    std::lock_guard<std::mutex> lock(inMutex_);
    // Decided under the lock stop() takes: a task is either queued before the
    // stop (and failed by failQueuedInProcess) or refused here, never left in a
    // queue nobody reads.
    stopping = inStop_ || inLoopExited_.load(std::memory_order_acquire);
    if (!stopping) {
      inQueue_.push_back(InProcessTask{std::move(request),
                                       std::move(onComplete),
                                       std::chrono::steady_clock::now()});
    }
  }
  if (stopping) {
    domain::video::VideoResponse refused(request.task_id);
    refused.error = "Server shutting down";
    if (onComplete) onComplete(std::move(refused));
    return;
  }
  inCv_.notify_one();
}

void VideoService::failQueuedInProcess(const std::string& error) {
  std::deque<InProcessTask> pending;
  {
    std::lock_guard<std::mutex> lock(inMutex_);
    pending.swap(inQueue_);
  }
  for (auto& task : pending) {
    domain::video::VideoResponse response(task.request.task_id);
    response.error = error;
    if (task.onComplete) task.onComplete(std::move(response));
  }
  refreshJobGauges();
}

domain::video::VideoResponse VideoService::runInProcessTask(
    const InProcessTask& task) {
  const uint32_t taskId = task.request.task_id;
  domain::video::VideoResponse response(taskId);
  const auto started = std::chrono::steady_clock::now();
  try {
    response.videoPath = inRunner_->run(task.request);
    faststartFinishedMp4(response.videoPath, taskId);
    if (response.videoPath.empty()) {
      response.error = "video response missing video_path";
    }
  } catch (const std::exception& e) {
    response.error = e.what();
    TT_LOG_ERROR("[VideoService] task {} failed: {}", taskId, e.what());
  } catch (...) {
    response.error = "unknown video runner error";
    TT_LOG_ERROR("[VideoService] task {} failed: unknown", taskId);
  }
  response.generationTimeSeconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();
  return response;
}

// The runner thread: creates and warms up the runner (the same factory a
// worker process uses), then takes tasks from the deque one at a time.
void VideoService::inProcessLoop() {
  try {
    // The interpreter (device runner) lives in the server process: keep the
    // server's own SIGINT / SIGTERM handlers.
    tt::runners::setVideoPythonSignalHandlers(false);
    using MediaRunner =
        tt::runners::IMediaRunner<domain::VideoGenerateRequest, std::string>;
    inRunner_ = tt::utils::RunnerRegistry::instance().createMedia<MediaRunner>(
        config::ModelService::VIDEO, videoConfig.runner_type,
        config::RunnerConfig{videoConfig});
    if (!inRunner_) {
      throw std::runtime_error("no media runner registered for runner_type=" +
                               config::toString(videoConfig.runner_type));
    }
    TT_LOG_INFO("[VideoService] In-process runner warming up ({})",
                inRunner_->runnerType());
    if (!inRunner_->warmup()) {
      throw std::runtime_error("runner warmup failed");
    }
  } catch (const std::exception& e) {
    const std::string reason =
        std::string("Video runner failed to start: ") + e.what();
    TT_LOG_ERROR("[VideoService] {}", reason);
    {
      std::lock_guard<std::mutex> lock(inFailureMutex_);
      inFailureReason_ = reason;
    }
    inFailed_.store(true, std::memory_order_release);
    failQueuedInProcess(reason);
    if (inRunner_) inRunner_->stop();
    releaseDeviceContextOnThisThread();
    inLoopExited_.store(true, std::memory_order_release);
    return;
  }
  inReady_.store(true, std::memory_order_release);
  TT_LOG_INFO("[VideoService] In-process runner ready");

  while (true) {
    std::optional<InProcessTask> taken;
    bool cancelledBeforeClaim = false;
    {
      std::unique_lock<std::mutex> lock(inMutex_);
      inCv_.wait(lock, [this] { return inStop_ || !inQueue_.empty(); });
      if (inStop_) break;
      taken.emplace(std::move(inQueue_.front()));
      inQueue_.pop_front();
      // Under the lock, so cancelJob() sees either "still in the deque" or
      // "in progress", never a task that is neither. A job cancelled between
      // jobs.create() and the enqueue is not in the deque for cancelJob() to
      // remove; the claim refuses it here instead.
      if (jobs.claimForRun(taken->request.task_id, 0,
                           static_cast<int>(::getpid()))) {
        inRunning_ = taken->request.task_id;
      } else {
        cancelledBeforeClaim = true;
      }
    }
    InProcessTask& task = *taken;
    if (cancelledBeforeClaim) {
      TT_LOG_INFO(
          "[VideoService] Skipping task {}: its job was cancelled before it "
          "was queued",
          task.request.task_id);
      refreshJobGauges();
      continue;
    }
    // Nothing a task does may end the loop: an exception that left this thread
    // would reach std::terminate and take the whole server down. The runner's
    // own errors are already turned into a failed response; this covers the
    // callbacks and the metrics around it.
    try {
      serveInProcessTask(task);
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[VideoService] task {}: unexpected error after the run: {}",
                   task.request.task_id, e.what());
    } catch (...) {
      TT_LOG_ERROR("[VideoService] task {}: unexpected error after the run",
                   task.request.task_id);
    }
    {
      std::lock_guard<std::mutex> lock(inMutex_);
      inRunning_.reset();
    }
    refreshJobGauges();
  }
  // Stop requested: close the device here, on the thread that opened it.
  try {
    inRunner_->stop();
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[VideoService] Stopping the runner failed: {}", e.what());
  } catch (...) {
    TT_LOG_ERROR("[VideoService] Stopping the runner failed");
  }
  releaseDeviceContextOnThisThread();
  inLoopExited_.store(true, std::memory_order_release);
}

// One claimed task: run it, record the metrics, hand the response to its
// callback.
void VideoService::serveInProcessTask(InProcessTask& task) {
  auto& metrics = tt::metrics::ServerMetrics::instance();
  const auto claimed = std::chrono::steady_clock::now();
  metrics.onVideoTaskClaimed(
      std::chrono::duration<double>(claimed - task.submitTime).count());
  refreshJobGauges();

  domain::video::VideoResponse response = runInProcessTask(task);

  const auto done = std::chrono::steady_clock::now();
  metrics.onVideoTaskDone(
      response.error.empty(), response.generationTimeSeconds,
      std::chrono::duration<double>(done - task.submitTime).count());
  // inRunning_ stays set until the callback has recorded the result (the loop
  // clears it), so the queue size does not drop while the job is in progress.
  if (task.onComplete) task.onComplete(std::move(response));
  metrics.onVideoOverhead(
      tt::metrics::ServerMetrics::VideoOverheadPhase::ResponseHandling,
      std::chrono::duration<double>(std::chrono::steady_clock::now() - done)
          .count());
}

}  // namespace tt::services
