// SPDX-License-Identifier: Apache-2.0
#include "utils/id_generator.hpp"
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "api/embedding_controller.hpp"

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "config/settings.hpp"
#include "profiling/tracy.hpp"
#include "services/base_service.hpp"
#include "utils/logger.hpp"
#include "utils/service_container.hpp"

namespace tt::api {

namespace {
// Simple thread pool for handling response callbacks
class CallbackThreadPool {
 public:
  CallbackThreadPool(size_t numThreads = 8) : stop(false) {
    for (size_t i = 0; i < numThreads; ++i) {
      workers.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock lock(mutex);
            cv.wait(lock, [this] { return stop || !tasks.empty(); });
            if (stop && tasks.empty()) return;
            task = std::move(tasks.front());
            tasks.pop();
          }
          task();
        }
      });
    }
  }

  ~CallbackThreadPool() {
    {
      std::lock_guard lock(mutex);
      stop = true;
    }
    cv.notify_all();
    for (auto& worker : workers) {
      if (worker.joinable()) worker.join();
    }
  }

  void submit(std::function<void()> task) {
    {
      std::lock_guard lock(mutex);
      tasks.push(std::move(task));
    }
    cv.notify_one();
  }

 private:
  std::vector<std::thread> workers;
  std::queue<std::function<void()>> tasks;
  TRACY_LOCKABLE(std::mutex, mutex);
  std::condition_variable_any cv;
  bool stop;
};

// Global thread pool for callbacks
CallbackThreadPool& getCallbackPool() {
  static CallbackThreadPool pool(16);  // 16 threads for handling callbacks
  return pool;
}
}  // namespace

EmbeddingController::EmbeddingController() {
  if (!tt::config::isEmbeddingService()) {
    return;
  }

  service_ = tt::utils::ServiceContainer::instance().embedding();
  if (!service_) {
    throw std::runtime_error(
        "[EmbeddingController] Embedding service not found in container. "
        "Ensure initializeServices() is called before Drogon starts.");
  }
  TT_LOG_INFO("[EmbeddingController] Initialized (service already started)");
}

EmbeddingController::~EmbeddingController() = default;

void EmbeddingController::createEmbedding(
    const drogon::HttpRequestPtr& req,
    std::function<void(const drogon::HttpResponsePtr&)>&& callback) {
  auto startTime = std::chrono::steady_clock::now();

  // Parse request body
  auto json = req->getJsonObject();
  if (!json) {
    auto resp = drogon::HttpResponse::newHttpJsonResponse(
        Json::Value("Invalid JSON body"));
    resp->setStatusCode(drogon::k400BadRequest);
    callback(resp);
    return;
  }

  // Validate required fields
  if (!json->isMember("input")) {
    Json::Value error;
    error["error"]["message"] = "Missing required field: input";
    error["error"]["type"] = "invalid_request_error";
    auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
    resp->setStatusCode(drogon::k400BadRequest);
    callback(resp);
    return;
  }

  // Build request
  uint32_t taskId = tt::utils::TaskIDGenerator::generate();
  domain::EmbeddingRequest request =
      domain::EmbeddingRequest::fromJson(*json, std::move(taskId));

  // Default model if not specified
  if (request.model.empty()) {
    request.model = "BAAI/bge-large-en-v1.5";
  }

  uint64_t reqNum = request_counter_.fetch_add(1);

  auto submitTime = std::chrono::steady_clock::now();

  getCallbackPool().submit([service = service_, request = std::move(request),
                            callback = std::move(callback), reqNum, startTime,
                            submitTime]() {
    try {
      auto response = service->submitRequest(std::move(request));
      auto gotResponseTime = std::chrono::steady_clock::now();

      if (!response.error.empty()) {
        Json::Value error;
        error["error"]["message"] = response.error;
        error["error"]["type"] = "server_error";
        auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
        resp->setStatusCode(drogon::k500InternalServerError);
        callback(resp);
        return;
      }

      Json::Value jsonResponse = response.toOpenaiJson();
      auto builtJsonTime = std::chrono::steady_clock::now();

      auto resp = drogon::HttpResponse::newHttpJsonResponse(jsonResponse);

      if (reqNum % 100 == 0) {
        double parseMs =
            std::chrono::duration<double, std::milli>(submitTime - startTime)
                .count();
        double waitMs = std::chrono::duration<double, std::milli>(
                            gotResponseTime - submitTime)
                            .count();
        double buildMs = std::chrono::duration<double, std::milli>(
                             builtJsonTime - gotResponseTime)
                             .count();
        double totalMs =
            std::chrono::duration<double, std::milli>(builtJsonTime - startTime)
                .count();
        TT_LOG_DEBUG(
            "[EmbeddingController] req={} parse={}ms wait={}ms build={}ms "
            "total={}ms",
            reqNum, parseMs, waitMs, buildMs, totalMs);
      }

      callback(resp);

    } catch (const services::QueueFullException& e) {
      Json::Value error;
      error["error"]["message"] = e.what();
      error["error"]["type"] = "rate_limit_exceeded";
      auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
      resp->setStatusCode(drogon::k429TooManyRequests);
      callback(resp);
    } catch (const std::exception& e) {
      Json::Value error;
      error["error"]["message"] = std::string("Internal error: ") + e.what();
      error["error"]["type"] = "server_error";
      auto resp = drogon::HttpResponse::newHttpJsonResponse(error);
      resp->setStatusCode(drogon::k500InternalServerError);
      callback(resp);
    }
  });
}

}  // namespace tt::api
