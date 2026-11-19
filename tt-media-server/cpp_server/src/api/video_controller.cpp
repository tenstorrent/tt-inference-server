// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/video_controller.hpp"

#include <json/json.h>

#include <filesystem>
#include <iomanip>
#include <optional>
#include <sstream>

#include "config/settings.hpp"
#include "domain/video/ordered_json.hpp"
#include "domain/video/video_mode.hpp"
#include "domain/video_generate_request.hpp"
#include "services/service_container.hpp"
#include "utils/http_range.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::api {

namespace {

using Callback = std::function<void(const drogon::HttpResponsePtr&)>;

drogon::HttpResponsePtr jsonBody(const Json::Value& value,
                                 drogon::HttpStatusCode status) {
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(status);
  resp->setContentTypeCode(drogon::CT_APPLICATION_JSON);
  resp->setBody(tt::domain::video::orderedJson(value));
  return resp;
}

// Python FastAPI puts the message in `detail`. The OpenAI `error` object this
// server already returns stays after it.
drogon::HttpResponsePtr videoError(drogon::HttpStatusCode status,
                                   const std::string& message,
                                   const std::string& type) {
  Json::Value body;
  body["detail"] = message;
  body["error"]["message"] = message;
  body["error"]["type"] = type;
  body["error"]["param"] = Json::nullValue;
  body["error"]["code"] = Json::nullValue;
  return jsonBody(body, status);
}

std::string generationTimeHeader(double seconds) {
  std::ostringstream out;
  out << std::fixed << std::setprecision(2) << seconds;
  return out.str();
}

// Same shape as Python's synchronous FileResponse: the body is the mp4, the
// filename is video_<task_id>.mp4, and X-Generation-Time carries the seconds.
drogon::HttpResponsePtr syncFileResponse(
    const domain::video::VideoResponse& response) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (response.videoPath.empty()) {
    return videoError(drogon::k500InternalServerError,
                      "Video generation failed: invalid file path returned",
                      "server_error");
  }
  if (!fs::is_regular_file(response.videoPath, ec)) {
    return videoError(
        drogon::k500InternalServerError,
        "Video generation failed: file not found at " + response.videoPath,
        "server_error");
  }
  const auto size = fs::file_size(response.videoPath, ec);
  if (ec || size == 0) {
    return videoError(drogon::k500InternalServerError,
                      "Video generation failed: empty file generated",
                      "server_error");
  }
  const std::string filename =
      "video_" + std::to_string(response.task_id) + ".mp4";
  auto resp = drogon::HttpResponse::newFileResponse(
      response.videoPath, filename, drogon::CT_CUSTOM, "video/mp4");
  resp->addHeader("X-Generation-Time",
                  generationTimeHeader(response.generationTimeSeconds));
  return resp;
}

void jobNotFound(const Callback& callback) {
  callback(videoError(drogon::k404NotFound, "Video job not found",
                      "invalid_request_error"));
}

}  // namespace

VideoController::VideoController() {
  if (!tt::config::isVideoService()) {
    return;
  }
  service = std::dynamic_pointer_cast<tt::services::VideoService>(
      tt::services::ServiceContainer::instance().getService(
          tt::config::ModelService::VIDEO));
  if (!service) {
    throw std::runtime_error(
        "[VideoController] Video service not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  TT_LOG_INFO("[VideoController] Initialized");
}

VideoController::~VideoController() = default;

void VideoController::generate(const drogon::HttpRequestPtr& req,
                               Callback&& callback) {
  auto json = req->getJsonObject();
  if (!json) {
    // 422 like FastAPI/pydantic for every malformed or invalid body.
    callback(videoError(drogon::k422UnprocessableEntity, "Invalid JSON body",
                        "invalid_request_error"));
    return;
  }
  if (!service->isModelReady()) {
    auto resp = videoError(
        drogon::k503ServiceUnavailable,
        "Model is still warming up. Poll /tt-liveness for model_ready=true.",
        "service_unavailable");
    resp->addHeader("Retry-After", "30");
    callback(resp);
    return;
  }

  // The mp4 is written only after the whole generation: refuse now instead of
  // spending a full run on a job that cannot save its result.
  if (const std::string dirError = service->outputDirError();
      !dirError.empty()) {
    TT_LOG_ERROR("[VideoController] generations refused: {}", dirError);
    callback(videoError(drogon::k503ServiceUnavailable,
                        "Video output directory is not writable: " + dirError,
                        "service_unavailable"));
    return;
  }

  std::optional<domain::VideoGenerateRequest> request;
  try {
    request = domain::VideoGenerateRequest::fromJson(
        *json, tt::utils::TaskIDGenerator::generate());
  } catch (const std::exception& e) {
    callback(videoError(drogon::k422UnprocessableEntity, e.what(),
                        "invalid_request_error"));
    return;
  }

  const uint64_t reqNum = requestCounter.fetch_add(1);
  bool asyncMode = true;
  try {
    asyncMode = tt::domain::video::videoRequestIsAsync(
        req->getParameter("async"), *json);
  } catch (const std::invalid_argument& e) {
    callback(videoError(drogon::k422UnprocessableEntity, e.what(),
                        "invalid_request_error"));
    return;
  }
  if (!asyncMode) {
    TT_LOG_INFO("[VideoController] generations req={} sync", reqNum);
    auto callbackPtr = std::make_shared<Callback>(std::move(callback));
    try {
      service->submitRequestAsync(
          std::move(*request),
          [callbackPtr](domain::video::VideoResponse response) {
            if (!response.error.empty()) {
              (*callbackPtr)(videoError(drogon::k500InternalServerError,
                                        response.error, "server_error"));
              return;
            }
            (*callbackPtr)(syncFileResponse(response));
          });
    } catch (const tt::services::QueueFullException& e) {
      auto resp = videoError(drogon::k429TooManyRequests, e.what(),
                             "rate_limit_exceeded");
      resp->addHeader("Retry-After", "60");
      (*callbackPtr)(resp);
    } catch (const std::exception& e) {
      TT_LOG_ERROR("[VideoController] generations req={} sync threw: {}",
                   reqNum, e.what());
      (*callbackPtr)(videoError(drogon::k500InternalServerError, e.what(),
                                "server_error"));
    }
    return;
  }

  try {
    auto job = service->createJob(std::move(*request));
    TT_LOG_INFO("[VideoController] generations req={} job={} queued", reqNum,
                job["id"].asString());
    callback(jsonBody(job, drogon::k202Accepted));
  } catch (const tt::services::QueueFullException& e) {
    auto resp = videoError(drogon::k429TooManyRequests, e.what(),
                           "rate_limit_exceeded");
    resp->addHeader("Retry-After", "60");
    callback(resp);
  } catch (const tt::services::MaxJobsReachedException& e) {
    auto resp = videoError(drogon::k503ServiceUnavailable, e.what(),
                           "service_unavailable");
    resp->addHeader("Retry-After", "60");
    callback(resp);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[VideoController] generations req={} threw: {}", reqNum,
                 e.what());
    callback(
        videoError(drogon::k500InternalServerError, e.what(), "server_error"));
  }
}

void VideoController::getJob(const drogon::HttpRequestPtr& /*req*/,
                             Callback&& callback, std::string jobId) {
  auto id = service->resolveJobId(jobId);
  auto job = id ? service->getJob(*id) : std::nullopt;
  if (!job) return jobNotFound(callback);
  callback(jsonBody(*job, drogon::k200OK));
}

void VideoController::listJobs(const drogon::HttpRequestPtr& /*req*/,
                               Callback&& callback) {
  callback(jsonBody(service->listJobs(), drogon::k200OK));
}

void VideoController::download(const drogon::HttpRequestPtr& req,
                               Callback&& callback, std::string jobId) {
  auto id = service->resolveJobId(jobId);
  auto job = id ? service->getJob(*id) : std::nullopt;
  if (!job) return jobNotFound(callback);  // no such job: 404

  // The job exists but has nothing to download yet (or ever): 409 with the
  // status, so a client knows to keep polling instead of giving up. Python
  // answers 404 for this case too.
  const std::string status = (*job)["status"].asString();
  if (status != "completed") {
    callback(videoError(drogon::k409Conflict,
                        "Video job is not completed (status: " + status + ")",
                        "invalid_request_error"));
    return;
  }

  auto path = service->jobResultPath(*id);
  std::error_code ec;
  if (!path || !std::filesystem::is_regular_file(*path, ec)) {
    callback(videoError(drogon::k404NotFound, "Video content not available",
                        "invalid_request_error"));
    return;
  }

  // The mp4 is already faststart (the worker remuxes it once, when the video
  // is finished), so it is streamed from disk. A single `Range` is honoured
  // (206 + Content-Range) so players can seek; other forms are ignored and
  // the whole file is sent.
  const std::string filename = std::filesystem::path(*path).filename().string();
  const uint64_t size = std::filesystem::file_size(*path, ec);
  const auto range =
      tt::utils::parseByteRange(req->getHeader("range"), ec ? 0 : size);

  drogon::HttpResponsePtr resp;
  switch (range.kind) {
    case tt::utils::ByteRange::Kind::Unsatisfiable:
      resp = videoError(drogon::k416RequestedRangeNotSatisfiable,
                        "Requested range not satisfiable",
                        "invalid_request_error");
      resp->addHeader("Content-Range", "bytes */" + std::to_string(size));
      break;
    case tt::utils::ByteRange::Kind::Partial:
      resp = drogon::HttpResponse::newFileResponse(
          *path, static_cast<size_t>(range.offset),
          static_cast<size_t>(range.length), true, filename, drogon::CT_CUSTOM,
          "video/mp4");
      resp->setStatusCode(drogon::k206PartialContent);
      break;
    case tt::utils::ByteRange::Kind::None:
      resp = drogon::HttpResponse::newFileResponse(
          *path, filename, drogon::CT_CUSTOM, "video/mp4");
      break;
  }
  resp->addHeader("Accept-Ranges", "bytes");
  callback(resp);
}

void VideoController::cancelJob(const drogon::HttpRequestPtr& /*req*/,
                                Callback&& callback, std::string jobId) {
  auto id = service->resolveJobId(jobId);
  auto job = id ? service->cancelJob(*id) : std::nullopt;
  if (!job) return jobNotFound(callback);
  callback(jsonBody(*job, drogon::k200OK));
}

void VideoController::deleteJob(const drogon::HttpRequestPtr& /*req*/,
                                Callback&& callback, std::string jobId) {
  auto id = service->resolveJobId(jobId);
  try {
    if (!id || !service->deleteJob(*id)) return jobNotFound(callback);
  } catch (const tt::services::JobStillActiveException& e) {
    callback(
        videoError(drogon::k409Conflict, e.what(), "invalid_request_error"));
    return;
  }
  Json::Value body;
  body["id"] = jobId;
  body["object"] = "video";
  body["deleted"] = true;
  callback(jsonBody(body, drogon::k200OK));
}

}  // namespace tt::api
