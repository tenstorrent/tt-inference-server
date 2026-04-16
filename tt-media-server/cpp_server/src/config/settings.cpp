// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#include "config/settings.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "config/defaults.hpp"
#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "utils/tokenizers/tokenizer.hpp"

namespace tt::config {

namespace {

/** Convert string to lowercase for case-insensitive environment variable
 * parsing. */
std::string toLower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s;
}

std::string envString(const char* name, const std::string& defaultValue) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : defaultValue;
}

/** Read env string and convert to lowercase for case-insensitive parsing. */
std::string envStringLower(const char* name, const std::string& defaultValue) {
  return toLower(envString(name, defaultValue));
}

unsigned long envUlong(const char* name, unsigned long defaultValue) {
  const char* v = std::getenv(name);
  if (!v || !*v) return defaultValue;
  try {
    return std::stoul(v);
  } catch (const std::exception&) {
    return defaultValue;
  }
}

/** Parse DEVICE_IDS like Python: "(0,1,2,3),(4,5,6,7)" -> ["0,1,2,3",
 * "4,5,6,7"]. */
std::vector<std::string> parseDeviceIds(const std::string& raw) {
  std::string s;
  for (char c : raw) {
    if (c != ' ') s += c;
  }
  if (s.empty()) {
    return {""};  // DEVICE_IDS="" means one worker, visible devices empty (use
                  // all).
  }
  std::vector<std::string> out;
  const std::string sep = "),(";
  size_t pos = 0;
  for (;;) {
    size_t next = s.find(sep, pos);
    std::string segment =
        (next == std::string::npos) ? s.substr(pos) : s.substr(pos, next - pos);
    if (!segment.empty()) {
      if (segment.front() == '(') segment.erase(0, 1);
      if (segment.back() == ')') segment.pop_back();
    }
    out.push_back(std::move(segment));
    if (next == std::string::npos) break;
    pos = next + sep.size();
  }
  return out;
}

const std::vector<std::string>& deviceIdsParsed() {
  static const std::vector<std::string> cached =
      parseDeviceIds(envString("DEVICE_IDS", defaults::DEVICE_IDS));
  return cached;
}

}  // namespace

ModelService modelService() {
  static const ModelService cached = modelServiceFromString(
      envStringLower("MODEL_SERVICE", defaults::MODEL_SERVICE));
  return cached;
}

bool isEmbeddingService() { return modelService() == ModelService::EMBEDDING; }

bool isLlmServiceEnabled() { return modelService() == ModelService::LLM; }

std::string runnerType() { return toString(modelService()); }

size_t numWorkers() { return deviceIdsParsed().size(); }

unsigned batchTimeoutMs() {
  return static_cast<unsigned>(
      envUlong("MAX_BATCH_DELAY_TIME_MS", defaults::MAX_BATCH_DELAY_TIME_MS));
}

std::string pythonPath() {
  return envString("TT_PYTHON_PATH", defaults::TT_PYTHON_PATH);
}

static std::filesystem::path tokenizersDir() {
  std::error_code ec;
  std::filesystem::path exePath =
      std::filesystem::read_symlink("/proc/self/exe", ec);
  if (!ec) {
    std::filesystem::path dir =
        exePath.parent_path().parent_path() / "tokenizers";
    if (std::filesystem::is_directory(dir)) {
      return dir;
    }
  }
  return {};
}

std::string tokenizerPath(ModelType model) {
  auto base = tokenizersDir();
  if (base.empty()) return "";
  std::string modelDir = utils::tokenizers::tokenizerDirForModel(model);
  std::filesystem::path p = base / modelDir / "tokenizer.json";
  if (std::filesystem::exists(p)) {
    return std::filesystem::absolute(p).string();
  }
  return "";
}

std::string tokenizerPath() { return tokenizerPath(modelType()); }

std::string tokenizerConfigPath(ModelType model) {
  auto base = tokenizersDir();
  if (base.empty()) return "";
  std::string modelDir = utils::tokenizers::tokenizerDirForModel(model);
  std::filesystem::path p = base / modelDir / "tokenizer_config.json";
  if (std::filesystem::exists(p)) {
    return std::filesystem::absolute(p).string();
  }
  return "";
}

std::string tokenizerConfigPath() { return tokenizerConfigPath(modelType()); }

std::string visibleDevicesForWorker(size_t workerIndex) {
  const auto& ids = deviceIdsParsed();
  if (workerIndex < ids.size()) return ids[workerIndex];
  return "";
}

std::string h2dSocketId() {
  static const std::string cached =
      envString("H2D_SOCKET_ID", defaults::H2D_SOCKET_ID);
  return cached;
}

std::string d2hSocketId() {
  static const std::string cached =
      envString("D2H_SOCKET_ID", defaults::D2H_SOCKET_ID);
  return cached;
}

unsigned pmConnectTimeoutMs() {
  return static_cast<unsigned>(
      envUlong("PM_CONNECT_TIMEOUT_MS", defaults::PM_CONNECT_TIMEOUT_MS));
}

size_t pmMaxUsers() {
  return static_cast<size_t>(envUlong("PM_MAX_USERS", defaults::PM_MAX_USERS));
}

bool useDeepseekMdFormat() {
  return static_cast<bool>(
      envUlong("USE_DEEPSEEK_MD_FORMAT", defaults::USE_DEEPSEEK_MD_FORMAT));
}

std::string ttTaskQueueName() {
  return envString("TT_TASK_QUEUE", defaults::TT_TASK_QUEUE);
}

std::string ttResultQueueName() {
  return envString("TT_RESULT_QUEUE", defaults::TT_RESULT_QUEUE);
}

std::string ttCancelQueueName() {
  return envString("TT_CANCEL_QUEUE", defaults::TT_CANCEL_QUEUE);
}

std::string ttWarmupSignalsQueueName() {
  return envString("TT_WARMUP_SIGNALS_QUEUE",
                   defaults::TT_WARMUP_SIGNALS_QUEUE);
}

std::string ttMemoryRequestQueueName() {
  return envString("TT_MEMORY_REQUEST_QUEUE",
                   defaults::TT_MEMORY_REQUEST_QUEUE);
}

std::string ttMemoryResultQueueName() {
  return envString("TT_MEMORY_RESULT_QUEUE", defaults::TT_MEMORY_RESULT_QUEUE);
}

LLMConfig llmEngineConfig() {
  static const LLMConfig cached = [] {
    LLMConfig cfg;
    cfg.stop_token_ids = utils::tokenizers::activeTokenizer().stopTokenIds();
    cfg.max_in_flight_count = maxInFlightCount();
    std::string backend =
        envStringLower("LLM_DEVICE_BACKEND", defaults::LLM_DEVICE_BACKEND);
    if (backend == "pipeline") {
      cfg.runner_type = ModelRunnerType::PIPELINE;
      cfg.max_in_flight_count = 1;
    } else if (backend == "prefill") {
      cfg.runner_type = ModelRunnerType::PREFILL;
      cfg.max_in_flight_count = 1;
    } else if (backend == "llama") {
      cfg.kvcache_block_size = 32;
      cfg.max_num_batched_tokens = 16384;
      cfg.runner_type = ModelRunnerType::LLAMA;
    } else if (backend == "mock") {
      cfg.runner_type = ModelRunnerType::MOCK;
    } else if (backend == "mock_pipeline") {
      cfg.runner_type = ModelRunnerType::MOCK_PIPELINE;
    } else if (backend == "pipeline_manager") {
      cfg.runner_type = ModelRunnerType::PIPELINE_MANAGER;
    } else {
      cfg.runner_type = ModelRunnerType::MOCK_PIPELINE;
    }
    cfg.scheduling_policy = schedulingPolicy();
    return cfg;
  }();
  return cached;
}

ModelType modelType() {
  static const ModelType cached = modelTypeFromDeviceBackend(
      envStringLower("LLM_DEVICE_BACKEND", defaults::LLM_DEVICE_BACKEND));
  return cached;
}

LLMMode llmMode() {
  static const LLMMode cached =
      llmModeFromString(envStringLower("LLM_MODE", defaults::LLM_MODE));
  return cached;
}

SchedulingPolicy schedulingPolicy() {
  static const SchedulingPolicy cached = schedulingPolicyFromString(
      envStringLower("SCHEDULING_POLICY", defaults::SCHEDULING_POLICY));
  return cached;
}

size_t maxInFlightCount() {
  static const size_t cached = static_cast<size_t>(
      envUlong("MAX_IN_FLIGHT_COUNT", defaults::MAX_IN_FLIGHT_COUNT));
  return cached;
}

std::string socketHost() {
  static const std::string cached =
      envString("SOCKET_HOST", defaults::SOCKET_HOST);
  return cached;
}

bool enableAccumulatedStreaming() {
  return envUlong("ENABLE_ACCUMULATED_STREAMING",
                  defaults::ENABLE_ACCUMULATED_STREAMING);
}

size_t maxAccumulatedTokens() {
  return static_cast<size_t>(
      envUlong("MAX_ACCUMULATED_TOKENS", defaults::MAX_ACCUMULATED_TOKENS));
}

uint16_t socketPort() {
  static const uint16_t cached =
      static_cast<uint16_t>(envUlong("SOCKET_PORT", defaults::SOCKET_PORT));
  return cached;
}

size_t maxQueueSize() {
  static const size_t cached =
      static_cast<size_t>(envUlong("MAX_QUEUE_SIZE", defaults::MAX_QUEUE_SIZE));
  return cached;
}

size_t maxSessionsCount() {
  return static_cast<size_t>(
      envUlong("MAX_SESSIONS_COUNT", defaults::MAX_SESSIONS_COUNT));
}

unsigned sessionEvictionRate() {
  return static_cast<unsigned>(
      envUlong("SESSION_EVICTION_RATE", defaults::SESSION_EVICTION_RATE));
}

size_t sessionEvictionCount() {
  return static_cast<size_t>(
      envUlong("SESSION_EVICTION_COUNT", defaults::SESSION_EVICTION_COUNT));
}

size_t maxTokensToPrefillOnDecode() {
  return static_cast<size_t>(
      envUlong("MAX_TOKENS_TO_PREFILL_ON_DECODE",
               defaults::MAX_TOKENS_TO_PREFILL_ON_DECODE));
}

std::string kafkaBrokers() {
  return envString("KAFKA_BROKERS", defaults::KAFKA_BROKERS);
}

std::string kafkaOffloadTopicName() {
  return envString("KAFKA_OFFLOAD_TOPIC_NAME",
                   defaults::KAFKA_OFFLOAD_TOPIC_NAME);
}

std::string kafkaGroupId() {
  return envString("KAFKA_GROUP_ID", defaults::KAFKA_GROUP_ID);
}
unsigned sessionAllocationMaxRetries() {
  return static_cast<unsigned>(
      envUlong("SESSION_ALLOCATION_MAX_RETRIES",
               defaults::SESSION_ALLOCATION_MAX_RETRIES));
}

}  // namespace tt::config
