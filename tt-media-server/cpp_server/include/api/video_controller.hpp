// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <drogon/HttpController.h>

#include <atomic>
#include <memory>

#include "services/video_service.hpp"

namespace tt::api {

/** Text-to-video controller. Both Python modes are live on this process.
 *
 * Async (`USE_ASYNC_VIDEO` unset or true, or `?async=true`): POST returns 202
 * and a job id. The client polls GET /generations/{id}, then downloads the mp4.
 * Sync (`USE_ASYNC_VIDEO=false` or `?async=false`): POST stays open and the
 * body is the mp4 file, same as Python `FileResponse`. The file also stays on
 * disk. i2v and ref2va are
 * later steps. SyncAdvice 404s these routes unless MODEL_SERVICE is video. */
class VideoController : public drogon::HttpController<VideoController> {
 public:
  // A route added here must also be registered in registerVideo()
  // (src/services/model_service_registration.cpp, `routes.registerRoute`), with
  // the path parameter written as `{job_id}`. Otherwise SyncAdvice rejects it
  // with 404 `route_not_found` and the controller is never reached.
  METHOD_LIST_BEGIN
  ADD_METHOD_TO(VideoController::generate, "/v1/videos/generations",
                drogon::Post);
  ADD_METHOD_TO(VideoController::getJob, "/v1/videos/generations/{job_id}",
                drogon::Get);
  ADD_METHOD_TO(VideoController::listJobs, "/v1/videos/jobs", drogon::Get);
  ADD_METHOD_TO(VideoController::download,
                "/v1/videos/generations/{job_id}/download", drogon::Get);
  ADD_METHOD_TO(VideoController::cancelJob,
                "/v1/videos/generations/{job_id}/cancel", drogon::Post);
  ADD_METHOD_TO(VideoController::deleteJob, "/v1/videos/generations/{job_id}",
                drogon::Delete);
  METHOD_LIST_END

  VideoController();
  ~VideoController();

  void generate(const drogon::HttpRequestPtr& req,
                std::function<void(const drogon::HttpResponsePtr&)>&& callback);
  void getJob(const drogon::HttpRequestPtr& req,
              std::function<void(const drogon::HttpResponsePtr&)>&& callback,
              std::string jobId);
  void listJobs(const drogon::HttpRequestPtr& req,
                std::function<void(const drogon::HttpResponsePtr&)>&& callback);
  void download(const drogon::HttpRequestPtr& req,
                std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                std::string jobId);
  void cancelJob(const drogon::HttpRequestPtr& req,
                 std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                 std::string jobId);
  void deleteJob(const drogon::HttpRequestPtr& req,
                 std::function<void(const drogon::HttpResponsePtr&)>&& callback,
                 std::string jobId);

 private:
  std::shared_ptr<services::VideoService> service;
  std::atomic<uint64_t> requestCounter{0};
};

}  // namespace tt::api
