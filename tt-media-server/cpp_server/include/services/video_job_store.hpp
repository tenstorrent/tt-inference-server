// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "domain/video/video_job.hpp"
#include "utils/uuid.hpp"
#include "utils/video_output_dir.hpp"

namespace tt::services {

class MaxJobsReachedException : public std::runtime_error {
 public:
  MaxJobsReachedException() : std::runtime_error("Maximum job limit reached") {}
};

class JobStillActiveException : public std::logic_error {
 public:
  JobStillActiveException()
      : std::logic_error(
            "Job is still active; cancel it first and delete once it reaches "
            "a terminal state") {}
};

class ForceDeleteNotAllowedException : public std::logic_error {
 public:
  ForceDeleteNotAllowedException(const std::string& jobType,
                                 const std::string& status)
      : std::logic_error("Only active training jobs can be force-deleted; a " +
                         jobType + " job is " + status) {}
};

/**
 * What differs per job type (Python: `AUTO_CLEANUP_EXEMPT_JOB_TYPES`,
 * `_validate_result_path_for_deletion` with `adapters_root` /
 * `merged_models_root`, and force delete of active training jobs).
 */
struct JobTypePolicy {
  bool autoCleanup = true;            // retention / sweep may evict it
  std::filesystem::path resultRoot;   // deletions only inside it; empty = any
  bool resultMayBeDirectory = false;  // result is a directory (training) and
                                      // is removed recursively
  bool forceDeletable = false;        // DELETE of an active job is allowed; the
                                      // caller must stop the worker
};

/**
 * Thread-safe in-memory job table (the C++ counterpart of Python JobManager).
 * Written by the Drogon I/O threads (submit/cancel/delete/poll) and by the
 * service's result-consumer threads (claim/complete), so every access takes
 * the mutex. Job records are small; the mp4 itself lives on disk.
 */
class VideoJobStore {
 public:
  VideoJobStore(size_t maxJobs, int64_t retentionSeconds)
      : maxJobs_(maxJobs), retentionSeconds_(retentionSeconds) {
    // The same rules as Python. Result roots of training / adapter merge are
    // not known here; the code that serves them sets them.
    policies_[domain::video::kJobTypeVideo] = JobTypePolicy{};
    policies_[domain::video::kJobTypeTraining] =
        JobTypePolicy{false, {}, true, true};
    policies_[domain::video::kJobTypeAdapterMerge] =
        JobTypePolicy{false, {}, true, false};
  }

  /** Overrides the rules of one job type (e.g. its result root). */
  void setJobTypePolicy(const std::string& jobType, JobTypePolicy policy) {
    std::lock_guard<std::mutex> lock(mutex_);
    policies_[jobType] = std::move(policy);
  }

  /**
   * Limits file deletion to this directory (Python validates the result path
   * before deleting). Unset = no restriction (unit tests). A path outside the
   * root is never deleted, only logged.
   */
  void setOutputRoot(std::filesystem::path root) {
    std::lock_guard<std::mutex> lock(mutex_);
    policies_[domain::video::kJobTypeVideo].resultRoot = std::move(root);
  }

  /** Registers a new QUEUED job. Throws MaxJobsReachedException when full. */
  Json::Value create(uint32_t taskId, std::string model,
                     Json::Value requestParameters,
                     std::string jobType = domain::video::kJobTypeVideo,
                     std::string orgId = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    evictExpiredLocked();
    if (jobs_.size() >= maxJobs_) throw MaxJobsReachedException{};
    domain::video::VideoJob job;
    job.taskId = taskId;
    job.jobType = std::move(jobType);
    job.orgId = std::move(orgId);
    job.uuid = utils::generateUuidV4();
    job.model = std::move(model);
    job.requestParameters = std::move(requestParameters);
    job.createdAt = domain::video::nowSeconds();
    auto json = job.toPublicJson();
    byUuid_.emplace(job.uuid, taskId);
    jobs_.emplace(taskId, std::move(job));
    return json;
  }

  /** Public job id (UUID) -> internal task id. nullopt = unknown / malformed.
   */
  std::optional<uint32_t> resolve(const std::string& publicId) const {
    if (!utils::isUuid(publicId)) return std::nullopt;
    std::string key = publicId;
    for (char& c : key)
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = byUuid_.find(key);
    if (it == byUuid_.end()) return std::nullopt;
    return it->second;
  }

  /** Jobs that are not terminal (Python: outstanding jobs vs max_queue_size).
   */
  size_t activeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<size_t>(
        std::count_if(jobs_.begin(), jobs_.end(), [](const auto& kv) {
          return !domain::video::isTerminal(kv.second.status);
        }));
  }

  /** Jobs per status, indexed like `JobStatus` (queued, in_progress,
   *  completed, failed, cancelled, cancelling). For the telemetry gauges. */
  std::array<size_t, 6> countByStatus() const {
    std::array<size_t, 6> counts{};
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& kv : jobs_) {
      ++counts[static_cast<size_t>(kv.second.status)];
    }
    return counts;
  }

  /** Drops a job that never reached the worker (submit threw). */
  void erase(uint32_t taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end()) return;
    byUuid_.erase(it->second.uuid);
    jobs_.erase(it);
  }

  /**
   * Worker picked the task up. A cancel that raced in stays CANCELLING.
   * `workerIdx`/`pid` record which worker holds it (Python
   * `worker_assignment.identity`); -1 = not known.
   */
  void markInProgress(uint32_t taskId, int workerIdx = -1, int pid = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end()) return;
    if (workerIdx >= 0) {
      it->second.worker = domain::video::WorkerIdentity{workerIdx, pid};
    }
    if (it->second.status == domain::video::JobStatus::QUEUED) {
      it->second.status = domain::video::JobStatus::IN_PROGRESS;
      it->second.progressAt = domain::video::nowSeconds();
    }
  }

  /**
   * The in-process runner takes a task: false when its job was cancelled (or
   * failed by the sweep) before the task reached the queue, so the task must
   * not run. A job that is queued becomes in progress; a task without a job
   * record (a synchronous request) always runs.
   */
  bool claimForRun(uint32_t taskId, int workerIdx, int pid) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end()) return true;
    if (it->second.status != domain::video::JobStatus::QUEUED) {
      return !domain::video::isTerminal(it->second.status);
    }
    it->second.worker = domain::video::WorkerIdentity{workerIdx, pid};
    it->second.status = domain::video::JobStatus::IN_PROGRESS;
    it->second.progressAt = domain::video::nowSeconds();
    return true;
  }

  /** The worker that holds the job, if one claimed it and it is not done. */
  std::optional<domain::video::WorkerIdentity> assignedWorker(
      uint32_t taskId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end()) return std::nullopt;
    return it->second.worker;
  }

  /**
   * Records that a running job is still making progress (Python
   * `Job.touch_progress`; training calls it for every metric). Without it a
   * long job would be failed by sweep() after the stuck limit.
   */
  void reportProgress(uint32_t taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it != jobs_.end()) it->second.progressAt = domain::video::nowSeconds();
  }

  /** Metrics / logs / checkpoints of a job (Python `job_metrics`, ...). */
  bool appendMetric(uint32_t taskId, Json::Value entry) {
    return appendTo(taskId, &domain::video::VideoJob::metrics,
                    std::move(entry));
  }
  bool appendLog(uint32_t taskId, Json::Value entry) {
    return appendTo(taskId, &domain::video::VideoJob::logs, std::move(entry));
  }
  bool appendCheckpoint(uint32_t taskId, Json::Value entry) {
    return appendTo(taskId, &domain::video::VideoJob::checkpoints,
                    std::move(entry));
  }
  std::optional<Json::Value> metrics(uint32_t taskId) const {
    return arrayOf(taskId, &domain::video::VideoJob::metrics);
  }
  std::optional<Json::Value> logs(uint32_t taskId) const {
    return arrayOf(taskId, &domain::video::VideoJob::logs);
  }
  std::optional<Json::Value> checkpoints(uint32_t taskId) const {
    return arrayOf(taskId, &domain::video::VideoJob::checkpoints);
  }

  struct SweepResult {
    size_t evicted = 0;
    size_t stuck = 0;
  };

  /**
   * Periodic cleanup (Python `_cleanup_old_jobs`): drops terminal jobs older
   * than the retention, and marks in-progress / cancelling jobs that made no
   * progress for `maxStuckSeconds` as failed (`stale_job`). A stuck job's
   * worker cannot be interrupted from here; if its result arrives later it is
   * discarded by complete(). `maxStuckSeconds <= 0` disables stuck detection.
   */
  SweepResult sweep(int64_t maxStuckSeconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    SweepResult result;
    const size_t before = jobs_.size();
    evictExpiredLocked();
    result.evicted = before - jobs_.size();
    if (maxStuckSeconds <= 0) return result;
    const int64_t cutoff = domain::video::nowSeconds() - maxStuckSeconds;
    for (auto& [id, job] : jobs_) {
      const bool running =
          job.status == domain::video::JobStatus::IN_PROGRESS ||
          job.status == domain::video::JobStatus::CANCELLING;
      if (running && job.progressAt != 0 && job.progressAt < cutoff) {
        job.status = domain::video::JobStatus::FAILED;
        job.errorCode = "stale_job";
        job.errorMessage =
            "Job made no progress and was force-cancelled by cleanup";
        job.completedAt = domain::video::nowSeconds();
        ++result.stuck;
      }
    }
    return result;
  }

  /**
   * Final result from the worker. If the job was cancelled meanwhile the
   * result is discarded (and the produced file removed).
   */
  void complete(uint32_t taskId, const std::string& videoPath,
                const std::string& error) {
    std::string orphanFile;
    std::string orphanType = domain::video::kJobTypeVideo;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = jobs_.find(taskId);
      if (it == jobs_.end()) {
        orphanFile = videoPath;  // job was deleted while running
      } else {
        auto& job = it->second;
        orphanType = job.jobType;
        job.worker.reset();
        if (job.status == domain::video::JobStatus::FAILED &&
            job.errorCode == "stale_job") {
          orphanFile = videoPath;  // already failed by sweep(); drop the result
          job.completedAt =
              job.completedAt ? job.completedAt : domain::video::nowSeconds();
        } else if (job.status == domain::video::JobStatus::CANCELLING ||
                   job.status == domain::video::JobStatus::CANCELLED) {
          job.completedAt = domain::video::nowSeconds();
          job.status = domain::video::JobStatus::CANCELLED;
          orphanFile = videoPath;
        } else if (!error.empty()) {
          job.completedAt = domain::video::nowSeconds();
          job.status = domain::video::JobStatus::FAILED;
          job.errorCode = "processing_error";
          job.errorMessage = error;
        } else {
          job.completedAt = domain::video::nowSeconds();
          job.status = domain::video::JobStatus::COMPLETED;
          job.resultPath = videoPath;
        }
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    removeResultLocked(orphanType, orphanFile);
  }

  /** `orgId` non-empty: only a job of that organisation is visible (Python
   *  `_get_job_if_authorized`). Empty: no filtering. Same for the calls below.
   */
  std::optional<Json::Value> get(uint32_t taskId,
                                 const std::string& orgId = {}) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end() || !visibleTo(it->second, orgId)) return std::nullopt;
    return it->second.toPublicJson();
  }

  Json::Value list(const std::string& orgId = {}) const {
    std::lock_guard<std::mutex> lock(mutex_);
    Json::Value arr(Json::arrayValue);
    for (const auto& [id, job] : jobs_) {
      if (visibleTo(job, orgId)) arr.append(job.toPublicJson());
    }
    return arr;
  }

  /** Path of the mp4 only for COMPLETED jobs (Python get_job_result_path). */
  std::optional<std::string> resultPath(uint32_t taskId,
                                        const std::string& orgId = {}) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end() || !visibleTo(it->second, orgId) ||
        it->second.status != domain::video::JobStatus::COMPLETED ||
        it->second.resultPath.empty()) {
      return std::nullopt;
    }
    return it->second.resultPath;
  }

  /**
   * Cancels a non-terminal job. nullopt = unknown or already terminal.
   * The device cannot be interrupted mid-denoise, so a running job becomes
   * CANCELLING and turns CANCELLED once the worker returns (result dropped).
   * A still-queued job is CANCELLED right away; its worker result, when it
   * eventually arrives, is dropped the same way.
   */
  std::optional<Json::Value> cancel(uint32_t taskId,
                                    const std::string& orgId = {}) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end() || !visibleTo(it->second, orgId) ||
        domain::video::isTerminal(it->second.status) ||
        it->second.status == domain::video::JobStatus::CANCELLING) {
      return std::nullopt;
    }
    auto& job = it->second;
    if (job.status == domain::video::JobStatus::QUEUED) {
      job.status = domain::video::JobStatus::CANCELLED;
      job.completedAt = domain::video::nowSeconds();
    } else {
      job.status = domain::video::JobStatus::CANCELLING;
    }
    return job.toPublicJson();
  }

  /**
   * Removes a terminal job and its result. Returns false if unknown; throws
   * JobStillActiveException while queued / running / cancelling. A late
   * worker result for an already removed job is dropped by complete().
   *
   * `force`: remove an active job too (Python: only active training jobs).
   * Throws ForceDeleteNotAllowedException for a type that does not allow it.
   * The record is removed; stopping or replacing its worker is the caller's
   * job (read `assignedWorker()` first).
   */
  bool remove(uint32_t taskId, const std::string& orgId = {},
              bool force = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end() || !visibleTo(it->second, orgId)) return false;
    const auto& job = it->second;
    if (!domain::video::isTerminal(job.status)) {
      if (!force) throw JobStillActiveException{};
      if (!policyFor(job.jobType).forceDeletable) {
        throw ForceDeleteNotAllowedException(
            job.jobType, domain::video::toString(job.status));
      }
    }
    removeResultLocked(job.jobType, job.resultPath);
    byUuid_.erase(job.uuid);
    jobs_.erase(it);
    return true;
  }

 private:
  void evictExpiredLocked() {
    if (retentionSeconds_ <= 0) return;
    const int64_t cutoff = domain::video::nowSeconds() - retentionSeconds_;
    for (auto it = jobs_.begin(); it != jobs_.end();) {
      const auto& job = it->second;
      if (domain::video::isTerminal(job.status) && job.completedAt != 0 &&
          job.completedAt < cutoff && policyFor(job.jobType).autoCleanup) {
        removeResultLocked(job.jobType, job.resultPath);
        byUuid_.erase(job.uuid);
        it = jobs_.erase(it);
      } else {
        ++it;
      }
    }
  }

  const JobTypePolicy& policyFor(const std::string& jobType) const {
    static const JobTypePolicy kDefault{};
    auto it = policies_.find(jobType);
    return it == policies_.end() ? kDefault : it->second;
  }

  static bool visibleTo(const domain::video::VideoJob& job,
                        const std::string& orgId) {
    return orgId.empty() || job.orgId == orgId;
  }

  using VectorMember = std::vector<Json::Value> domain::video::VideoJob::*;

  bool appendTo(uint32_t taskId, VectorMember member, Json::Value entry) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end()) return false;
    (it->second.*member).push_back(std::move(entry));
    return true;
  }

  std::optional<Json::Value> arrayOf(uint32_t taskId,
                                     VectorMember member) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end()) return std::nullopt;
    Json::Value arr(Json::arrayValue);
    for (const auto& entry : it->second.*member) arr.append(entry);
    return arr;
  }

  /**
   * Deletes a job's result like Python `_delete_result_path`: only inside the
   * type's result root (when one is set), and a directory only for types whose
   * result is a directory. Caller holds mutex_.
   */
  void removeResultLocked(const std::string& jobType,
                          const std::string& path) const {
    if (path.empty() || path.rfind("mock://", 0) == 0) return;
    const JobTypePolicy& policy = policyFor(jobType);
    if (!policy.resultRoot.empty() &&
        !utils::isInsideDir(path, policy.resultRoot)) {
      std::fprintf(stderr,
                   "[VideoJobStore] Refusing to delete '%s': outside result "
                   "dir '%s'\n",
                   path.c_str(), policy.resultRoot.string().c_str());
      return;
    }
    std::error_code ec;
    if (policy.resultMayBeDirectory &&
        std::filesystem::is_directory(path, ec)) {
      std::filesystem::remove_all(path, ec);
    } else {
      std::filesystem::remove(path, ec);
    }
  }

  mutable std::mutex mutex_;
  std::map<uint32_t, domain::video::VideoJob> jobs_;
  std::map<std::string, uint32_t> byUuid_;
  size_t maxJobs_;
  int64_t retentionSeconds_;
  std::map<std::string, JobTypePolicy> policies_;
};

}  // namespace tt::services
