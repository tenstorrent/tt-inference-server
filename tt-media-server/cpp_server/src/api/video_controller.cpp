// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "api/video_controller.hpp"

#include <drogon/MultiPart.h>
#include <json/json.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <optional>
#include <sstream>

#include "config/settings.hpp"
#include "domain/video/video_deployment.hpp"
#include "domain/video/video_media.hpp"
#include "domain/video/video_mode.hpp"
#include "domain/video_generate_request.hpp"
#include "services/service_container.hpp"
#include "utils/http_range.hpp"
#include "utils/id_generator.hpp"
#include "utils/image_codec.hpp"
#include "utils/logger.hpp"

namespace tt::api {

namespace {

using Callback = std::function<void(const drogon::HttpResponsePtr&)>;

drogon::HttpResponsePtr jsonBody(const Json::Value& value,
                                 drogon::HttpStatusCode status) {
  auto resp = drogon::HttpResponse::newHttpJsonResponse(value);
  resp->setStatusCode(status);
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
// A synchronous request has no job record, so nothing would ever delete its
// file: the bytes are read into the response and the file is discarded here
// (also when the client has already gone away).
drogon::HttpResponsePtr syncFileResponse(
    const domain::video::VideoResponse& response,
    const std::function<void(const std::string&)>& discard) {
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
  std::string bytes(static_cast<size_t>(size), '\0');
  {
    std::ifstream in(response.videoPath, std::ios::binary);
    if (!in.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
      return videoError(
          drogon::k500InternalServerError,
          "Video generation failed: cannot read " + response.videoPath,
          "server_error");
    }
  }
  discard(response.videoPath);
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setBody(std::move(bytes));
  resp->setContentTypeCodeAndCustomString(drogon::CT_CUSTOM, "video/mp4");
  resp->addHeader("Content-Disposition", "attachment; filename=" + filename);
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
  handle(req, std::move(callback), domain::video::VideoRoute::Text, nullptr);
}

void VideoController::generateI2V(const drogon::HttpRequestPtr& req,
                                  Callback&& callback) {
  handle(req, std::move(callback), domain::video::VideoRoute::I2V, nullptr);
}

namespace {

/** "png", "jpeg" or "webp" from the magic bytes; empty for anything else. */
std::string imageFormat(const std::string& content) {
  auto at = [&](size_t i) { return static_cast<uint8_t>(content[i]); };
  if (content.size() >= 8 && at(0) == 0x89 && content.compare(1, 3, "PNG") == 0)
    return "png";
  if (content.size() >= 3 && at(0) == 0xFF && at(1) == 0xD8 && at(2) == 0xFF)
    return "jpeg";
  if (content.size() >= 12 && content.compare(0, 4, "RIFF") == 0 &&
      content.compare(8, 4, "WEBP") == 0)
    return "webp";
  return "";
}

/** A refused request. The 4xx statuses are Python's. */
drogon::HttpResponsePtr refusal(int status, const std::string& message) {
  const auto code = static_cast<drogon::HttpStatusCode>(status);
  return videoError(code, message,
                    status >= 500 ? "server_error" : "invalid_request_error");
}

/** URLs are refused with 400 unless the operator enables downloads: with the
 *  Python default (no allowed domains, no public hosts) every URL-valued media
 *  field is refused the same way. Downloading is a separate step. */
std::optional<std::string> mediaUrlRefusal(const Json::Value& body) {
  const char* message =
      "Media URL downloads are not enabled on this server; send the media "
      "as base64";
  const Json::Value& prompts = body["image_prompts"];
  if (prompts.isArray()) {
    for (const auto& entry : prompts) {
      if (entry.isObject() && entry["image"].isString() &&
          domain::video::isMediaUrl(entry["image"].asString())) {
        return message;
      }
    }
  }
  return std::nullopt;
}

}  // namespace

void VideoController::generateI2VUpload(const drogon::HttpRequestPtr& req,
                                        Callback&& callback) {
  namespace dv = domain::video;
  if (const auto no = dv::routeRefusal(dv::VideoRoute::I2V, deployment_)) {
    callback(refusal(422, *no));
    return;
  }
  drogon::MultiPartParser parser;
  if (parser.parse(req) != 0 || parser.getFiles().empty()) {
    callback(refusal(422,
                     "Request must be multipart/form-data with an "
                     "`image` file and a `prompt` field"));
    return;
  }
  const auto& params = parser.getParameters();
  const auto prompt = params.find("prompt");
  const drogon::HttpFile* image = nullptr;
  for (const auto& file : parser.getFiles()) {
    if (file.getItemName() == "image") image = &file;
  }
  if (prompt == params.end() || image == nullptr) {
    callback(refusal(422, "Fields `prompt` and `image` are required"));
    return;
  }
  const std::string_view content = image->fileContent();
  if (content.size() > dv::kMaxUploadBytes) {
    callback(refusal(413, "Image exceeds " +
                              std::to_string(dv::kMaxUploadBytes) +
                              "-byte upload cap"));
    return;
  }
  const std::string format = imageFormat(std::string(content.substr(0, 16)));
  if (format.empty()) {
    callback(refusal(415,
                     "Unsupported image content_type; allowed: "
                     "['image/jpeg', 'image/png', 'image/webp']"));
    return;
  }
  Json::Value body;
  body["prompt"] = prompt->second;
  Json::Value entry;
  entry["image"] = tt::utils::image_codec::base64Encode(
      reinterpret_cast<const uint8_t*>(content.data()), content.size());
  try {
    auto number = [&](const char* name) -> std::optional<long long> {
      const auto it = params.find(name);
      if (it == params.end() || it->second.empty()) return std::nullopt;
      return std::stoll(it->second);
    };
    entry["frame_pos"] =
        static_cast<Json::Int64>(number("frame_pos").value_or(0));
    body["image_prompts"].append(entry);
    body["num_inference_steps"] =
        static_cast<Json::Int64>(number("num_inference_steps").value_or(12));
    if (const auto seed = number("seed")) {
      body["seed"] = static_cast<Json::Int64>(*seed);
    }
  } catch (const std::exception&) {
    callback(refusal(422,
                     "frame_pos, num_inference_steps and seed must be "
                     "integers"));
    return;
  }
  const auto negative = params.find("negative_prompt");
  if (negative != params.end() && !negative->second.empty()) {
    body["negative_prompt"] = negative->second;
  }
  handle(req, std::move(callback), dv::VideoRoute::I2V, &body);
}

void VideoController::handle(const drogon::HttpRequestPtr& req,
                             Callback&& callback,
                             domain::video::VideoRoute route,
                             const Json::Value* prepared) {
  namespace dv = domain::video;
  // The deployment decides which route it serves (422, Python's wording).
  if (const auto no = dv::routeRefusal(route, deployment_)) {
    callback(refusal(422, *no));
    return;
  }
  Json::Value body;
  if (prepared != nullptr) {
    body = *prepared;
  } else {
    auto json = req->getJsonObject();
    if (!json) {
      // 422 like FastAPI/pydantic for every malformed or invalid body.
      callback(videoError(drogon::k422UnprocessableEntity, "Invalid JSON body",
                          "invalid_request_error"));
      return;
    }
    body = *json;
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

  // Each route reads only its own media field, like the Python request
  // classes: `/generations` ignores image_prompts, `/i2v` needs them.
  if (const auto no = mediaUrlRefusal(body)) {
    callback(refusal(400, *no));
    return;
  }
  Json::Value shaped = body;
  if (route != dv::VideoRoute::I2V) shaped.removeMember("image_prompts");
  shaped.removeMember("references");
  if (route == dv::VideoRoute::I2V &&
      (!shaped.isMember("image_prompts") || shaped["image_prompts"].isNull())) {
    callback(refusal(422, "Missing required field: image_prompts"));
    return;
  }

  std::optional<domain::VideoGenerateRequest> request;
  try {
    request = domain::VideoGenerateRequest::fromJson(
        shaped, tt::utils::TaskIDGenerator::generate());
    dv::checkRequestMedia(*request, deployment_);
  } catch (const dv::MediaRequestError& e) {
    callback(refusal(e.status(), e.what()));
    return;
  } catch (const std::exception& e) {
    callback(videoError(drogon::k422UnprocessableEntity, e.what(),
                        "invalid_request_error"));
    return;
  }

  const uint64_t reqNum = requestCounter.fetch_add(1);
  bool asyncMode = true;
  try {
    asyncMode = tt::domain::video::videoRequestIsAsync(
        req->getParameter("async"), body);
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
          [callbackPtr, svc = service](domain::video::VideoResponse response) {
            if (!response.error.empty()) {
              (*callbackPtr)(videoError(drogon::k500InternalServerError,
                                        response.error, "server_error"));
              return;
            }
            // The same rule as a job's result: only a file inside the output
            // directory is served. A path outside it is not ours to read or
            // delete, so it is refused and left alone.
            if (!svc->isResultPathAllowed(response.videoPath)) {
              TT_LOG_ERROR(
                  "[VideoController] sync task {}: result '{}' is outside the "
                  "output dir",
                  response.task_id, response.videoPath);
              (*callbackPtr)(videoError(
                  drogon::k500InternalServerError,
                  "Video generation failed: result file is outside the "
                  "output directory",
                  "server_error"));
              return;
            }
            (*callbackPtr)(
                syncFileResponse(response, [svc](const std::string& path) {
                  svc->discardSyncResult(path);
                }));
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
  if (ec) {
    // The file went away between the check above and now (eviction, DELETE).
    callback(videoError(drogon::k404NotFound, "Video content not available",
                        "invalid_request_error"));
    return;
  }
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
