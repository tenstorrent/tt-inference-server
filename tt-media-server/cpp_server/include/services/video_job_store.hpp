// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
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

/** Too many jobs that are not finished (the video queue limit, HTTP 429). */
class ActiveJobLimitException : public std::runtime_error {
 public:
  ActiveJobLimitException()
      : std::runtime_error("Request queue is full, please retry later") {}
};

class JobStillActiveException : public std::logic_error {
 public:
  JobStillActiveException()
      : std::logic_error(
            "Job is still active; cancel it first and delete once it reaches "
            "a terminal state") {}
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
      : maxJobs_(maxJobs), retentionSeconds_(retentionSeconds) {}

  /**
   * Limits the result files to this directory (Python validates the result path
   * before deleting). A result outside it is refused by complete(), and a file
   * outside it is never deleted. Unset = no restriction (unit tests).
   */
  void setOutputRoot(std::filesystem::path root) {
    std::lock_guard<std::mutex> lock(mutex_);
    outputRoot_ = std::move(root);
  }

  /**
   * Registers a new QUEUED job. The limits are checked under the same lock that
   * inserts the job, so concurrent requests cannot all pass a check made before
   * any of them is stored. Same order as Python: more than `maxActive` jobs
   * that are not finished throws ActiveJobLimitException (0 = no limit), then a
   * full table throws MaxJobsReachedException.
   */
  Json::Value create(uint32_t taskId, std::string model,
                     Json::Value requestParameters, size_t maxActive = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    evictExpiredLocked();
    if (maxActive > 0 && activeCountLocked() >= maxActive) {
      throw ActiveJobLimitException{};
    }
    if (jobs_.size() >= maxJobs_) throw MaxJobsReachedException{};
    domain::video::VideoJob job;
    job.taskId = taskId;
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
    return activeCountLocked();
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
   */
  void markInProgress(uint32_t taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end()) return;
    if (it->second.status == domain::video::JobStatus::QUEUED) {
      it->second.status = domain::video::JobStatus::IN_PROGRESS;
      it->second.progressAt = domain::video::nowSeconds();
    }
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
   * result is discarded (and the produced file removed). A result path outside
   * the output root (see setOutputRoot) fails the job.
   */
  void complete(uint32_t taskId, const std::string& videoPath,
                const std::string& error) {
    std::string orphanFile;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = jobs_.find(taskId);
      if (it == jobs_.end()) {
        orphanFile = videoPath;  // job was deleted while running
      } else {
        auto& job = it->second;
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
        } else if (!resultPathAllowed(videoPath)) {
          // The runner reported a file outside the result root. Storing it
          // would let /download serve it, so the job fails instead. The file is
          // not touched: it is not ours to delete.
          std::fprintf(stderr,
                       "[VideoJobStore] Task %u: result '%s' is outside the "
                       "output dir\n",
                       taskId, videoPath.c_str());
          job.completedAt = domain::video::nowSeconds();
          job.status = domain::video::JobStatus::FAILED;
          job.errorCode = "processing_error";
          job.errorMessage = "Result file is outside the output directory";
        } else {
          job.completedAt = domain::video::nowSeconds();
          job.status = domain::video::JobStatus::COMPLETED;
          job.resultPath = videoPath;
        }
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    removeResultLocked(orphanFile);
  }

  /**
   * Whether a result file a runner reported may be served: the rule complete()
   * applies to a job, for a request that has no job (sync mode). Inside the
   * output root (when one is set); `mock://` results are not files.
   */
  bool isResultPathAllowed(const std::string& path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return resultPathAllowed(path);
  }

  std::optional<Json::Value> get(uint32_t taskId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end()) return std::nullopt;
    return it->second.toPublicJson();
  }

  Json::Value list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    Json::Value arr(Json::arrayValue);
    for (const auto& [id, job] : jobs_) {
      arr.append(job.toPublicJson());
    }
    return arr;
  }

  /** Path of the mp4 only for COMPLETED jobs (Python get_job_result_path). */
  std::optional<std::string> resultPath(uint32_t taskId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end() ||
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
  std::optional<Json::Value> cancel(uint32_t taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end() || domain::video::isTerminal(it->second.status) ||
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
   */
  bool remove(uint32_t taskId) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = jobs_.find(taskId);
    if (it == jobs_.end()) return false;
    const auto& job = it->second;
    if (!domain::video::isTerminal(job.status)) throw JobStillActiveException{};
    removeResultLocked(job.resultPath);
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
          job.completedAt < cutoff) {
        removeResultLocked(job.resultPath);
        byUuid_.erase(job.uuid);
        it = jobs_.erase(it);
      } else {
        ++it;
      }
    }
  }

  size_t activeCountLocked() const {
    return static_cast<size_t>(
        std::count_if(jobs_.begin(), jobs_.end(), [](const auto& kv) {
          return !domain::video::isTerminal(kv.second.status);
        }));
  }

  /** True when a result path may be stored: inside the output root (when one
   *  is set). `mock://` results are not files. Caller holds mutex_. */
  bool resultPathAllowed(const std::string& path) const {
    if (path.empty() || path.rfind("mock://", 0) == 0) return true;
    return outputRoot_.empty() || utils::isInsideDir(path, outputRoot_);
  }

  /**
   * Deletes a job's result file like Python `_delete_result_path`: only inside
   * the output root (when one is set). Caller holds mutex_.
   */
  void removeResultLocked(const std::string& path) const {
    if (path.empty() || path.rfind("mock://", 0) == 0) return;
    if (!outputRoot_.empty() && !utils::isInsideDir(path, outputRoot_)) {
      std::fprintf(stderr,
                   "[VideoJobStore] Refusing to delete '%s': outside result "
                   "dir '%s'\n",
                   path.c_str(), outputRoot_.string().c_str());
      return;
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  mutable std::mutex mutex_;
  std::map<uint32_t, domain::video::VideoJob> jobs_;
  std::map<std::string, uint32_t> byUuid_;
  size_t maxJobs_;
  int64_t retentionSeconds_;
  std::filesystem::path outputRoot_;  // empty = no restriction
};

}  // namespace tt::services
