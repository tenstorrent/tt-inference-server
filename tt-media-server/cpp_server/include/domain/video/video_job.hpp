// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <ctime>
#include <optional>
#include <string>
#include <vector>

namespace tt::domain::video {

/** Python `JobTypes` (config/constants.py). The store is generic over the job
 *  type so training can reuse it later; only "video" is served today. */
inline constexpr const char* kJobTypeVideo = "video";
inline constexpr const char* kJobTypeTraining = "training";
inline constexpr const char* kJobTypeAdapterMerge = "adapter_merge";

/** Which worker process holds a job (Python `Job.assigned_worker_identity`,
 *  `(worker_id, pid)`). Needed later to stop or replace that worker. */
struct WorkerIdentity {
  int workerIdx = -1;
  int pid = 0;
};

/** Same states as Python `JobStatus` (utils/job_manager.py). */
enum class JobStatus {
  QUEUED,
  IN_PROGRESS,
  COMPLETED,
  FAILED,
  CANCELLED,
  CANCELLING,
};

inline const char* toString(JobStatus status) {
  switch (status) {
    case JobStatus::QUEUED:
      return "queued";
    case JobStatus::IN_PROGRESS:
      return "in_progress";
    case JobStatus::COMPLETED:
      return "completed";
    case JobStatus::FAILED:
      return "failed";
    case JobStatus::CANCELLED:
      return "cancelled";
    case JobStatus::CANCELLING:
      return "cancelling";
  }
  return "unknown";
}

inline bool isTerminal(JobStatus status) {
  return status == JobStatus::COMPLETED || status == JobStatus::FAILED ||
         status == JobStatus::CANCELLED;
}

/** One async video job. The job id is the decimal task id. */
struct VideoJob {
  uint32_t taskId = 0;
  std::string jobType = kJobTypeVideo;
  std::string orgId;  // empty = none (Python `Job.org_id`)
  std::string uuid;   // public id; empty = use the decimal task id
  std::string model;
  Json::Value requestParameters;
  JobStatus status = JobStatus::QUEUED;
  int64_t createdAt = 0;
  int64_t completedAt = 0;  // 0 = not finished
  int64_t progressAt = 0;   // last time the worker claimed it; 0 = never
  std::string resultPath;
  std::string errorCode;
  std::string errorMessage;
  std::optional<WorkerIdentity> worker;  // set when a worker claims it
  // Python `job_metrics`, `job_logs`, `job_checkpoints` (used by training;
  // empty for video, so they cost nothing there).
  std::vector<Json::Value> metrics;
  std::vector<Json::Value> logs;
  std::vector<Json::Value> checkpoints;

  std::string id() const {
    return uuid.empty() ? std::to_string(taskId) : uuid;
  }

  /** Mirrors Python `Job.to_public_dict()`. */
  Json::Value toPublicJson() const {
    Json::Value json;
    json["id"] = id();
    json["job_type"] = jobType;
    json["status"] = toString(status);
    json["created_at"] = static_cast<Json::Int64>(createdAt);
    json["model"] = model;
    json["request_parameters"] = requestParameters;
    if (!orgId.empty()) json["org_id"] = orgId;
    if (completedAt != 0) {
      json["completed_at"] = static_cast<Json::Int64>(completedAt);
    }
    if (!errorMessage.empty()) {
      json["error"]["code"] = errorCode;
      json["error"]["message"] = errorMessage;
    }
    return json;
  }
};

inline int64_t nowSeconds() { return static_cast<int64_t>(std::time(nullptr)); }

}  // namespace tt::domain::video
