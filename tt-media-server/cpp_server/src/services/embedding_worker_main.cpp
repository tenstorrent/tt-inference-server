// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

#include "services/embedding_worker_main.hpp"

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "config/settings.hpp"
#include "config/types.hpp"
#include "domain/embedding_request.hpp"
#include "runtime/runners/i_embedding_runner.hpp"
#include "services/embedding_codec.hpp"
#include "services/embedding_pipe.hpp"
#include "utils/id_generator.hpp"
#include "utils/logger.hpp"

namespace tt::services::embedding_detail {

namespace {

/**
 * Export everything the runner reads from the environment, BEFORE it is
 * built: TT_VISIBLE_DEVICES scopes the worker to its chips, the thread caps
 * keep 32 workers from oversubscribing the host cores, and TT_METAL_CACHE
 * gives each worker a private kernel-cache directory (mirroring the Python
 * server's setup_runner_environment — sharing one directory makes concurrent
 * JIT compilation race). Also logs the full worker configuration.
 */
void exportWorkerEnvironment(int workerId,
                             const tt::config::EmbeddingConfig& cfg,
                             const std::string& visibleDevices) {
  setenv("TT_VISIBLE_DEVICES", visibleDevices.c_str(), 1);

  const char* cpuThreads = "2";
  const char* torchThreads = "1";
  setenv("OMP_NUM_THREADS", cpuThreads, 1);
  setenv("MKL_NUM_THREADS", cpuThreads, 1);
  setenv("TORCH_NUM_THREADS", torchThreads, 1);

  if (const char* throttle = std::getenv("DEFAULT_THROTTLE_LEVEL");
      throttle && *throttle) {
    setenv("TT_MM_THROTTLE_PERF", throttle, 1);
  }

  if (const char* metalHome = std::getenv("TT_METAL_HOME");
      metalHome && *metalHome) {
    std::string deviceSuffix = visibleDevices;
    std::replace(deviceSuffix.begin(), deviceSuffix.end(), ',', '_');
    const std::string metalCache =
        std::string(metalHome) + "/built/" + deviceSuffix;
    setenv("TT_METAL_CACHE", metalCache.c_str(), 1);
  }

  const char* metalCacheEnv = std::getenv("TT_METAL_CACHE");
  const char* throttleEnv = std::getenv("TT_MM_THROTTLE_PERF");
  TT_LOG_INFO(
      "[Worker {}] Started (PID {}, runner_type={}, TT_VISIBLE_DEVICES={}, "
      "TT_METAL_CACHE={}, DEVICE={}, max_batch_size={}, OMP_NUM_THREADS={}, "
      "TORCH_NUM_THREADS={}, TT_MM_THROTTLE_PERF={})",
      workerId, getpid(), tt::config::toString(cfg.runner_type), visibleDevices,
      metalCacheEnv ? metalCacheEnv : "(default)", cfg.device,
      cfg.max_batch_size, cpuThreads, torchThreads,
      throttleEnv ? throttleEnv : "(unset)");
}

/** Build the runner for this worker's devices, or exit the child process. */
std::unique_ptr<runners::IEmbeddingRunner> buildRunnerOrDie(
    int workerId, const tt::config::EmbeddingConfig& cfg,
    const std::string& visibleDevices) {
  try {
    auto workerCfg = cfg;
    workerCfg.worker_id = static_cast<size_t>(workerId);
    workerCfg.visible_devices = visibleDevices;
    return runners::makeEmbeddingRunner(workerCfg);
  } catch (const std::exception& e) {
    TT_LOG_ERROR("[Worker {}] Could not build runner: {}", workerId, e.what());
    _exit(1);
  }
}

/** Parse one length-prefixed JSON payload (object or array) into a request
 * batch; nullopt on malformed JSON. */
std::optional<std::vector<domain::EmbeddingRequest>> parseBatch(
    const std::string& requestJson, int workerId) {
  Json::Value reqJson;
  Json::CharReaderBuilder builder;
  std::istringstream iss(requestJson);
  std::string errors;
  if (!Json::parseFromStream(builder, iss, &reqJson, &errors)) {
    TT_LOG_ERROR("[Worker {}] Failed to parse request: {}", workerId, errors);
    return std::nullopt;
  }

  auto taskIdFromJson = [](const Json::Value& j) -> uint32_t {
    return (j.isMember("task_id") && j["task_id"].isUInt())
               ? j["task_id"].asUInt()
               : tt::utils::TaskIDGenerator::generate();
  };

  std::vector<domain::EmbeddingRequest> batch;
  if (reqJson.isArray()) {
    for (const auto& item : reqJson)
      batch.push_back(
          domain::EmbeddingRequest::fromJson(item, taskIdFromJson(item)));
  } else {
    batch.push_back(
        domain::EmbeddingRequest::fromJson(reqJson, taskIdFromJson(reqJson)));
  }
  return batch;
}

/** Serve loop: read a batch, run it, write the encoded responses. Returns
 * when the request pipe reports EOF (parent closed it — shutdown). */
void serveLoop(runners::IEmbeddingRunner& runner, int workerId, int readFd,
               int writeFd) {
  while (true) {
    std::string requestJson = pipeReadString(readFd);
    if (requestJson.empty()) break;

    auto batch = parseBatch(requestJson, workerId);
    if (!batch) continue;

    TT_LOG_INFO("[Worker {}] Processing batch of {} requests", workerId,
                batch->size());

    auto responses = runner.run(*batch);
    auto buf = embedding_codec::encodeResponses(*batch, responses);

    if (!pipeWrite(writeFd, buf.data(), buf.size())) {
      TT_LOG_ERROR("[Worker {}] Failed to write response", workerId);
    }
  }
}

}  // namespace

[[noreturn]] void workerProcessMain(int workerId, int readFd, int writeFd) {
  const size_t wid = static_cast<size_t>(workerId);
  const auto cfg = tt::config::embeddingEngineConfig();
  const std::string visibleDevices = tt::config::visibleDevicesForWorker(wid);

  exportWorkerEnvironment(workerId, cfg, visibleDevices);

  auto runner = buildRunnerOrDie(workerId, cfg, visibleDevices);

  if (!runner->warmup()) {
    TT_LOG_ERROR("[Worker {}] Warmup failed!", workerId);
    _exit(1);
  }

  // Tell the parent we can serve; until this arrives the parent keeps the
  // worker marked not-ready and won't dispatch to it.
  if (!pipeWrite(writeFd, WORKER_READY_SENTINEL,
                 sizeof(WORKER_READY_SENTINEL) - 1)) {
    TT_LOG_ERROR("[Worker {}] Failed to send ready signal", workerId);
    _exit(1);
  }
  TT_LOG_INFO("[Worker {}] Ready", workerId);

  serveLoop(*runner, workerId, readFd, writeFd);

  runner->close();
  _exit(0);
}

}  // namespace tt::services::embedding_detail
