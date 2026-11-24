// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <json/json.h>

#include <cstdint>
#include <cstring>
#include <ctime>
#include <optional>
#include <string>
#include <vector>

#include "domain/video_generate_request.hpp"

namespace tt::domain::video {

/** Python `JobTypes.VIDEO` (config/constants.py): the only job type served. */
inline constexpr const char* kJobTypeVideo = "video";

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
  std::string uuid;   // public id; empty = use the decimal task id
  std::string model;
  Json::Value requestParameters;
  JobStatus status = JobStatus::QUEUED;
  int64_t createdAt = 0;
  int64_t completedAt = 0;  // 0 = not finished
  // Last time the worker claimed it; 0 = never. It is not refreshed while the
  // model runs, so a healthy generation longer than JOB_MAX_STUCK_TIME_SECONDS
  // (3 h by default) is swept as stale_job. A runner heartbeat would fix that
  // (follow-up).
  int64_t progressAt = 0;
  std::string resultPath;
  std::string errorCode;
  std::string errorMessage;

  std::string id() const {
    return uuid.empty() ? std::to_string(taskId) : uuid;
  }

  /** Python `VideoGenerateRequest.model_dump()`: every field is present and an
   *  unset optional is null. The shape validator fills a missing
   *  num_inference_steps with the default (20) before the job is stored, so the
   *  default is applied here as well for a job that was built without it. Keys
   *  this server adds on top of the Python schema are kept. */
  Json::Value publicRequestParameters() const {
    static const char* const kFields[] = {
        "prompt",   "negative_prompt", "num_inference_steps", "seed",
        "aspect_ratio", "height",      "width",               "fps",
        "duration", "num_frames"};
    Json::Value out(Json::objectValue);
    for (const char* key : kFields) {
      const bool set = requestParameters.isObject() &&
                       requestParameters.isMember(key) &&
                       !requestParameters[key].isNull();
      if (set) {
        out[key] = requestParameters[key];
      } else if (std::strcmp(key, "num_inference_steps") == 0) {
        out[key] = kDefaultVideoInferenceSteps;
      } else {
        out[key] = Json::nullValue;
      }
    }
    if (requestParameters.isObject()) {
      for (const auto& name : requestParameters.getMemberNames()) {
        if (!out.isMember(name)) out[name] = requestParameters[name];
      }
    }
    return out;
  }

  /** Mirrors Python `Job.to_public_dict()`. */
  Json::Value toPublicJson() const {
    Json::Value json;
    json["id"] = id();
    json["job_type"] = kJobTypeVideo;
    json["status"] = toString(status);
    json["created_at"] = static_cast<Json::Int64>(createdAt);
    json["model"] = model;
    json["request_parameters"] = publicRequestParameters();
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
