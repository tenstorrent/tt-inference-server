// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "config/settings.hpp"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "config/defaults.hpp"
#include "config/runner_config.hpp"
#include "config/types.hpp"
#include "utils/logger.hpp"
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

std::string resolveBlazeSocketDescriptorPrefix() {
  switch (modelType()) {
    case ModelType::DEEPSEEK_R1_0528:
      return "deepseek";
    case ModelType::LLAMA_3_1_8B_INSTRUCT:
      return "llama";
    case ModelType::KIMI_K2_6:
    case ModelType::KIMI_K2_7_CODE:
      return "kimi";
    case ModelType::GPT_OSS_120B:
      return "gpt-oss";
    case ModelType::MINIMAX_M2_7:
      return "minimax";
    case ModelType::GLM_5_1:
    case ModelType::GLM_5_2:
      return "glm";
    case ModelType::DEEPSEEK_V4_PRO:
      return "deepseek";
  }
  throw std::runtime_error(
      "Unhandled MODEL_TYPE while resolving BLAZE_SOCKET_DESCRIPTOR_PREFIX");
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

bool envBool(const char* name, bool defaultValue) {
  const char* v = std::getenv(name);
  if (!v || !*v) return defaultValue;
  const std::string lower = toLower(v);
  if (lower == "1" || lower == "true" || lower == "yes" || lower == "on") {
    return true;
  }
  if (lower == "0" || lower == "false" || lower == "no" || lower == "off") {
    return false;
  }
  return defaultValue;
}

/** Read a KV cache block size env var; value must be divisible by 32. */
size_t kvCacheSizeFromEnv(const char* envName, size_t defaultValue) {
  size_t value = static_cast<size_t>(envUlong(envName, defaultValue));
  if (value % 32) {
    TT_LOG_WARN("[Config] {}={} is not divisible by 32, using default={}",
                envName, value, defaultValue);
    return defaultValue;
  }
  return value;
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

bool isLlmService() { return modelService() == ModelService::LLM; }

bool isImageService() { return modelService() == ModelService::IMAGE; }

std::string runnerType() { return toString(modelService()); }

size_t numWorkers() { return deviceIdsParsed().size(); }

size_t callbackPoolThreads() {
  static const size_t cached = [] {
    // CALLBACK_POOL_THREADS=N picks an explicit size; 0 or unset auto-scales.
    const size_t fromEnv =
        static_cast<size_t>(envUlong("CALLBACK_POOL_THREADS", 0));
    if (fromEnv > 0) {
      return std::min(fromEnv, defaults::CALLBACK_POOL_THREADS_MAX);
    }
    // Auto: at least the legacy floor, never less than the per-deploy worker
    // count so HTTP dispatch threads never silently cap below DEVICE_IDS
    // (the root cause of 16-wide serialization on 32-chip Galaxy SDXL).
    return std::min(
        std::max<size_t>(numWorkers(), defaults::CALLBACK_POOL_THREADS_MIN),
        defaults::CALLBACK_POOL_THREADS_MAX);
  }();
  return cached;
}

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
  std::filesystem::path jsonPath = base / modelDir / "tokenizer.json";
  if (std::filesystem::exists(jsonPath)) {
    return std::filesystem::absolute(jsonPath).string();
  }
  std::filesystem::path tiktokenPath = base / modelDir / "tiktoken.model";
  if (std::filesystem::exists(tiktokenPath)) {
    return std::filesystem::absolute(tiktokenPath).string();
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

std::string blazeSocketDescriptorPrefix() {
  static const std::string cached = envString(
      "BLAZE_SOCKET_DESCRIPTOR_PREFIX", resolveBlazeSocketDescriptorPrefix());
  return cached;
}

unsigned pmConnectTimeoutMs() {
  return static_cast<unsigned>(
      envUlong("PM_CONNECT_TIMEOUT_MS", defaults::PM_CONNECT_TIMEOUT_MS));
}

size_t pmMaxUsers() {
  return static_cast<size_t>(envUlong("PM_MAX_USERS", defaults::PM_MAX_USERS));
}

unsigned warmupTimeoutMs() {
  return static_cast<unsigned>(
      envUlong("WARMUP_TIMEOUT_MS", defaults::WARMUP_TIMEOUT_MS));
}

unsigned outputHangTimeoutMs() {
  return static_cast<unsigned>(
      envUlong("OUTPUT_HANG_TIMEOUT_MS", defaults::OUTPUT_HANG_TIMEOUT_MS));
}

std::string ttTaskQueueName() {
  return envString("TT_TASK_QUEUE", defaults::TT_TASK_QUEUE);
}

std::string wireFormat() {
  return envString("WIRE_FORMAT", defaults::WIRE_FORMAT);
}

std::string ttResultQueueName() {
  return envString("TT_RESULT_QUEUE", defaults::TT_RESULT_QUEUE);
}

std::string ttCancelQueueName() {
  return envString("TT_CANCEL_QUEUE", defaults::TT_CANCEL_QUEUE);
}

std::string ttMediaTaskQueueName() {
  return envString("TT_MEDIA_TASK_QUEUE", defaults::TT_MEDIA_TASK_QUEUE);
}

std::string ttMediaResultQueueName() {
  return envString("TT_MEDIA_RESULT_QUEUE", defaults::TT_MEDIA_RESULT_QUEUE);
}

std::string ttWarmupSignalsQueueName() {
  return envString("TT_WARMUP_SIGNALS_QUEUE",
                   defaults::TT_WARMUP_SIGNALS_QUEUE);
}

uint32_t modelNumLayers() {
  return envUlong("MODEL_NUM_LAYERS", defaults::MODEL_NUM_LAYERS);
}

uint32_t prefillChunkSize() {
  return envUlong("PREFILL_CHUNK_SIZE", defaults::PREFILL_CHUNK_SIZE);
}

std::string ttMemoryRequestQueueName() {
  return envString("TT_MEMORY_REQUEST_QUEUE",
                   defaults::TT_MEMORY_REQUEST_QUEUE);
}

std::string ttMemoryResultQueueName() {
  return envString("TT_MEMORY_RESULT_QUEUE", defaults::TT_MEMORY_RESULT_QUEUE);
}

std::string workerMetricsShmName() {
  return envString("TT_WORKER_METRICS_SHM", defaults::TT_WORKER_METRICS_SHM);
}

size_t resultQueueCapacity() {
  return envUlong("RESULT_QUEUE_CAPACITY", defaults::RESULT_QUEUE_CAPACITY);
}

size_t cancelQueueCapacity() {
  return envUlong("CANCEL_QUEUE_CAPACITY", defaults::CANCEL_QUEUE_CAPACITY);
}

size_t memoryQueueCapacity() {
  return envUlong("MEMORY_QUEUE_CAPACITY", defaults::MEMORY_QUEUE_CAPACITY);
}

bool useMockScheduler() {
  return envBool("MOCK_USE_SCHEDULER", defaults::MOCK_USE_SCHEDULER);
}

unsigned mockPrefillLatencyMs() {
  return static_cast<unsigned>(
      envUlong("MOCK_PREFILL_CHUNK_LATENCY_MS",
               defaults::MOCK_PREFILL_CHUNK_LATENCY_MS));
}

unsigned mockDecodeTokenLatencyUs() {
  return static_cast<unsigned>(envUlong(
      "MOCK_DECODE_TOKEN_LATENCY_US", defaults::MOCK_DECODE_TOKEN_LATENCY_US));
}

uint32_t mockDecodeTokenId() {
  return static_cast<uint32_t>(
      envUlong("MOCK_DECODE_TOKEN_ID", defaults::MOCK_DECODE_TOKEN_ID));
}

LLMConfig llmEngineConfig() {
  static const LLMConfig cached = [] {
    LLMConfig cfg;
    cfg.stop_token_ids = utils::tokenizers::staticInfo().stopTokenIds;
    cfg.max_in_flight_count = maxInFlightCount();
    std::string backend =
        envStringLower("LLM_DEVICE_BACKEND", defaults::LLM_DEVICE_BACKEND);
    if (backend == "llama") {
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

namespace {

/** Parse "WxH" (case-insensitive 'x'); std::nullopt if malformed or either
 *  dimension is non-positive. Strict: rejects trailing junk like
 * "1024x1024foo". */
std::optional<std::pair<size_t, size_t>> parseResolution(const std::string& s) {
  std::istringstream ss(s);
  long long w = 0, h = 0;
  char sep = 0;
  ss >> w >> sep >> h;
  if (!ss || w <= 0 || h <= 0 || (sep != 'x' && sep != 'X')) {
    return std::nullopt;
  }
  ss >> std::ws;
  if (!ss.eof()) {
    return std::nullopt;
  }
  return std::make_pair(static_cast<size_t>(w), static_cast<size_t>(h));
}

/** Parse "1,1" / "2,4" -> {1,1} / {2,4}. Empty/whitespace -> {1,1}. Rejects
 *  single-axis input ("8") with a clear error: silently promoting to {8, 1}
 *  would flip tensor parallelism on without the operator asking for it.
 *  Rejects more than 2 dims and trailing junk: the rest of the code indexes
 *  [0]/[1] and treats the mesh as (rows, cols). */
std::vector<size_t> parseMeshShape(const std::string& s) {
  std::vector<size_t> out;
  std::istringstream ss(s);
  size_t v = 0;
  while (ss >> v) {
    out.push_back(v);
    if (ss.peek() == ',') ss.ignore();
  }
  ss >> std::ws;
  if (!ss.eof()) {
    throw std::runtime_error(
        "[Config] DEVICE_MESH_SHAPE has trailing junk after parsing; "
        "expected 'rows,cols', got '" +
        s + "'");
  }
  if (out.empty()) return {1, 1};
  if (out.size() == 1) {
    throw std::runtime_error(
        "[Config] DEVICE_MESH_SHAPE must be 'rows,cols' (e.g. '1,1' for "
        "single device, '2,4' for 2x4 mesh); got '" +
        s + "'");
  }
  if (out.size() > 2) {
    throw std::runtime_error(
        "[Config] DEVICE_MESH_SHAPE must be 2-D 'rows,cols'; got '" + s +
        "' with " + std::to_string(out.size()) + " dimensions");
  }
  if (out[0] == 0 || out[1] == 0) {
    throw std::runtime_error(
        "[Config] DEVICE_MESH_SHAPE dimensions must be >= 1; got '" + s + "'");
  }
  return out;
}

/** Shared reader for every in-process media runner config. */
void readMediaRunnerConfig(MediaRunnerConfigBase& cfg) {
  cfg.max_batch_size = static_cast<size_t>(envUlong("MAX_BATCH_SIZE", 1));
  cfg.device_mesh_shape = parseMeshShape(envString("DEVICE_MESH_SHAPE", "1,1"));
  cfg.is_galaxy = envBool("IS_GALAXY", false);
  cfg.device = envStringLower("DEVICE", "");
  cfg.model_weights_path = envString("MODEL_WEIGHTS_PATH", "");
  cfg.weights_distribution_timeout_seconds = static_cast<unsigned>(
      envUlong("WEIGHTS_DISTRIBUTION_TIMEOUT_SECONDS", 1800));
  cfg.visible_devices = visibleDevicesForWorker(0);
}

}  // namespace

ImageConfig imageEngineConfig() {
  static const ImageConfig cached = [] {
    ImageConfig cfg;
    const std::string runner =
        envStringLower("MODEL_RUNNER_TYPE", "tt_sdxl_generate");
    if (runner == "tt_sdxl_generate") {
      cfg.runner_type = ModelRunnerType::TT_SDXL_GENERATE;
    } else if (runner == "tt_sdxl_image_to_image") {
      cfg.runner_type = ModelRunnerType::TT_SDXL_IMAGE_TO_IMAGE;
    } else if (runner == "tt_sdxl_edit") {
      cfg.runner_type = ModelRunnerType::TT_SDXL_EDIT;
    } else {
      throw std::runtime_error(
          "[Config] Unknown image MODEL_RUNNER_TYPE='" + runner +
          "'; expected one of: tt_sdxl_generate, tt_sdxl_image_to_image, "
          "tt_sdxl_edit");
    }

    readMediaRunnerConfig(cfg);

    if (auto wh = parseResolution(envString("SDXL_IMAGE_RESOLUTION", ""))) {
      cfg.image_width = wh->first;
      cfg.image_height = wh->second;
    }
    return cfg;
  }();
  return cached;
}

RunnerConfig workerRunnerConfig(size_t workerIndex) {
  switch (modelService()) {
    case ModelService::IMAGE: {
      auto cfg = imageEngineConfig();
      cfg.worker_id = workerIndex;
      cfg.visible_devices = visibleDevicesForWorker(workerIndex);
      return cfg;
    }
    case ModelService::EMBEDDING:
      return EmbeddingConfig{};
    case ModelService::LLM:
    default:
      return llmEngineConfig();
  }
}

ModelType modelType() {
  static const ModelType cached = [] {
    // Derive model type from MODEL env var
    std::string m = envString("MODEL", defaults::MODEL);
    if (m == "moonshotai/Kimi-K2.6") return ModelType::KIMI_K2_6;
    if (m == "moonshotai/Kimi-K2.7-Code") return ModelType::KIMI_K2_7_CODE;
    if (m == "meta-llama/Llama-3.1-8B-Instruct")
      return ModelType::LLAMA_3_1_8B_INSTRUCT;
    if (m == "openai/gpt-oss-120b") return ModelType::GPT_OSS_120B;
    if (m == "MiniMaxAI/MiniMax-M2.7") return ModelType::MINIMAX_M2_7;
    if (m == "zai-org/GLM-5.1") return ModelType::GLM_5_1;
    if (m == "zai-org/GLM-5.2") return ModelType::GLM_5_2;
    if (m == "deepseek-ai/DeepSeek-V4-Pro") return ModelType::DEEPSEEK_V4_PRO;
    return ModelType::DEEPSEEK_R1_0528;
  }();
  return cached;
}

Model model() {
  static const Model cached =
      modelFromString(envString("MODEL", defaults::MODEL));
  return cached;
}

bool sampleOnlyInReasoning() {
  switch (modelType()) {
    // DeepSeek samples only inside the reasoning phase; argmax otherwise.
    case ModelType::DEEPSEEK_R1_0528:
      return true;
    case ModelType::LLAMA_3_1_8B_INSTRUCT:
    case ModelType::KIMI_K2_6:
    case ModelType::KIMI_K2_7_CODE:
    case ModelType::GPT_OSS_120B:
    case ModelType::MINIMAX_M2_7:
    case ModelType::GLM_5_1:
    case ModelType::GLM_5_2:
    case ModelType::DEEPSEEK_V4_PRO:
      return false;
  }
  return false;
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

uint16_t socketPort() {
  static const uint16_t cached =
      static_cast<uint16_t>(envUlong("SOCKET_PORT", defaults::SOCKET_PORT));
  return cached;
}

bool usePrefillGateway() {
  return envUlong("USE_PREFILL_GATEWAY",
                  defaults::USE_PREFILL_GATEWAY ? 1UL : 0UL) != 0UL;
}

namespace {
std::string getHostname() {
  std::string host(256, '\0');
  if (::gethostname(host.data(), host.size()) != 0) {
    return "unknown-host";
  }
  host.resize(std::strlen(host.c_str()));
  return host;
}
}  // namespace

std::string prefillServerId() {
  static const std::string cached = [] {
    std::string v = envString("PREFILL_SERVER_ID", defaults::PREFILL_SERVER_ID);
    if (!v.empty()) return v;
    return getHostname() + ":" + std::to_string(socketPort());
  }();
  return cached;
}

std::string logInstanceTag(int workerIndex) {
  // Role distinguishes the node: LLM_MODE for the LLM service, else the service
  // name. Worker subprocesses keep the base role and append their index, so a
  // merged log still separates "decode" vs "prefill" nodes and their
  // "decode-worker0" / "prefill-worker0" workers.
  std::string role = isLlmService() ? std::string(toString(llmMode()))
                                    : std::string(toString(modelService()));
  if (workerIndex >= 0) {
    role += "-worker" + std::to_string(workerIndex);
  }
  return role;
}

uint32_t prefillMaxInFlight() {
  return static_cast<uint32_t>(
      envUlong("PREFILL_MAX_IN_FLIGHT", defaults::PREFILL_MAX_IN_FLIGHT));
}

size_t maxQueueSize() {
  static const size_t cached =
      static_cast<size_t>(envUlong("MAX_QUEUE_SIZE", defaults::MAX_QUEUE_SIZE));
  return cached;
}

namespace {
std::atomic<size_t> maxSessionsCountOverride{0};
}

size_t maxSessionsCount() {
  size_t overrideVal = maxSessionsCountOverride.load(std::memory_order_relaxed);
  if (overrideVal > 0) {
    return overrideVal;
  }
  return static_cast<size_t>(
      envUlong("MAX_SESSIONS_COUNT", defaults::MAX_SESSIONS_COUNT));
}

void setMaxSessionsCount(size_t count) {
  maxSessionsCountOverride.store(count, std::memory_order_relaxed);
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

size_t maxContextLength() {
  static const size_t cached = static_cast<size_t>(
      envUlong("MAX_CONTEXT_LENGTH", defaults::MAX_CONTEXT_LENGTH));
  return cached;
}

size_t maxISL() {
  static const size_t cached =
      static_cast<size_t>(envUlong("MAX_ISL", defaults::MAX_ISL));
  return cached;
}

size_t minTokensToCopy() {
  static const size_t cached = static_cast<size_t>(
      envUlong("MIN_TOKENS_TO_COPY", defaults::MIN_TOKENS_TO_COPY));
  return cached;
}

size_t kvCacheBlockSize() {
  static const size_t cached = []() {
    return kvCacheSizeFromEnv("KV_CACHE_BLOCK_SIZE",
                              defaults::KV_CACHE_BLOCK_SIZE);
  }();
  return cached;
}

size_t kvCacheFirstBlockSize() {
  static const size_t cached = []() {
    return kvCacheSizeFromEnv("KV_CACHE_FIRST_BLOCK_SIZE",
                              defaults::KV_CACHE_FIRST_BLOCK_SIZE);
  }();
  return cached;
}

float prefixCacheHitThreshold() {
  const unsigned long val = envUlong("PREFIX_CACHE_HIT_THRESHOLD",
                                     defaults::PREFIX_CACHE_HIT_THRESHOLD);
  if (val > 100) {
    TT_LOG_WARN(
        "[Config] PREFIX_CACHE_HIT_THRESHOLD={} out of range [0,100], "
        "using {}",
        val, defaults::PREFIX_CACHE_HIT_THRESHOLD);
    return static_cast<float>(defaults::PREFIX_CACHE_HIT_THRESHOLD);
  }
  return static_cast<float>(val);
}

bool useFastMode() {
  return envUlong("USE_FAST_MODE", defaults::USE_FAST_MODE);
}

bool enableMigration() {
  return envBool("ENABLE_MIGRATION", defaults::ENABLE_MIGRATION);
}

std::string migrationCmdQueueName() {
  return envString("MIGRATION_CMD_QUEUE_NAME",
                   defaults::MIGRATION_CMD_QUEUE_NAME);
}

std::string migrationTableQueueName() {
  return envString("MIGRATION_TABLE_QUEUE_NAME",
                   defaults::MIGRATION_TABLE_QUEUE_NAME);
}

std::string migrationRespQueueName() {
  return envString("MIGRATION_RESP_QUEUE_NAME",
                   defaults::MIGRATION_RESP_QUEUE_NAME);
}

std::string prefillAckChannelName() {
  return envString("PREFILL_ACK_CHANNEL_NAME",
                   defaults::PREFILL_ACK_CHANNEL_NAME);
}

std::string kafkaBrokers() {
  return envString("KAFKA_BROKERS", defaults::KAFKA_BROKERS);
}

std::string kafkaOffloadTopicName() {
  return envString("KAFKA_OFFLOAD_TOPIC_NAME",
                   defaults::KAFKA_OFFLOAD_TOPIC_NAME);
}

std::string kafkaMigrationRequestTopic() {
  return envString("KAFKA_MIGRATION_REQUEST_TOPIC",
                   defaults::KAFKA_MIGRATION_REQUEST_TOPIC);
}

std::string kafkaMigrationAckTopic() {
  return envString("KAFKA_MIGRATION_ACK_TOPIC",
                   defaults::KAFKA_MIGRATION_ACK_TOPIC);
}

uint32_t migrationPrefillEndpointId() {
  return static_cast<uint32_t>(
      envUlong("MIGRATION_PREFILL_ENDPOINT_ID",
               defaults::MIGRATION_PREFILL_ENDPOINT_ID));
}

uint32_t migrationDecodeEndpointId() {
  return static_cast<uint32_t>(envUlong(
      "MIGRATION_DECODE_ENDPOINT_ID", defaults::MIGRATION_DECODE_ENDPOINT_ID));
}

std::string kafkaGroupId() {
  return envString("KAFKA_GROUP_ID", defaults::KAFKA_GROUP_ID);
}
unsigned sessionAllocationMaxRetries() {
  return static_cast<unsigned>(
      envUlong("SESSION_ALLOCATION_MAX_RETRIES",
               defaults::SESSION_ALLOCATION_MAX_RETRIES));
}

bool dynamoEndpointEnabled() {
  return envBool("DYNAMO_ENDPOINT_ENABLED", defaults::DYNAMO_ENDPOINT_ENABLED);
}

std::string dynamoBindHost() {
  return envString("DYNAMO_BIND_HOST", defaults::DYNAMO_BIND_HOST);
}

std::string dynamoEtcdEndpoints() {
  // Prefer DYNAMO_ETCD_ENDPOINTS (cpp_server-specific). Fall back to
  // ETCD_ENDPOINTS — Dynamo's Rust runtime reads the same name, so a single
  // export propagates to both processes when start_dynamo.sh wires them
  // together.
  if (const char* v = std::getenv("DYNAMO_ETCD_ENDPOINTS"); v && *v) {
    return v;
  }
  if (const char* v = std::getenv("ETCD_ENDPOINTS"); v && *v) {
    return v;
  }
  return defaults::DYNAMO_ETCD_ENDPOINTS;
}

std::string specDecodeMode() {
  return envString("SPEC_DECODE_MODE", defaults::SPEC_DECODE_MODE);
}

size_t mtpLevel() {
  auto val = static_cast<size_t>(envUlong("MTP_LEVEL", defaults::MTP_LEVEL));
  if (val > 4) {
    throw std::runtime_error("MTP_LEVEL must be <= 4");
  }
  return val;
}

int64_t dynamoEtcdLeaseTtlSecs() {
  const char* v = std::getenv("DYNAMO_ETCD_LEASE_TTL_SECS");
  if (!v || !*v) return defaults::DYNAMO_ETCD_LEASE_TTL_SECS;
  try {
    return std::stoll(v);
  } catch (const std::exception&) {
    return defaults::DYNAMO_ETCD_LEASE_TTL_SECS;
  }
}

std::string dynamoNamespace() {
  return envString("DYNAMO_NAMESPACE", defaults::DYNAMO_NAMESPACE);
}

std::string dynamoComponent() {
  if (const char* v = std::getenv("DYNAMO_COMPONENT"); v && *v) {
    return v;
  }
  if (dynamoWorkerRole() == "prefill") {
    return "prefill";
  }
  return envString("DYNAMO_COMPONENT", defaults::DYNAMO_COMPONENT);
}

std::string dynamoEndpointName() {
  return envString("DYNAMO_ENDPOINT_NAME", defaults::DYNAMO_ENDPOINT_NAME);
}

std::string dynamoWorkerRole() {
  std::string role =
      envString("DYNAMO_WORKER_ROLE", defaults::DYNAMO_WORKER_ROLE);
  if (role == "prefill" || role == "decode") {
    return role;
  }
  return llmMode() == LLMMode::PREFILL_ONLY ? "prefill" : "decode";
}

bool dynamoNativePrefillHandoffEnabled() {
  return envBool("DYNAMO_NATIVE_PREFILL_HANDOFF_ENABLED",
                 defaults::DYNAMO_NATIVE_PREFILL_HANDOFF_ENABLED);
}

bool dynamoDecodeOrchestratesPrefill() {
  return envBool("DYNAMO_DECODE_ORCHESTRATES_PREFILL",
                 defaults::DYNAMO_DECODE_ORCHESTRATES_PREFILL);
}

bool dynamoPrefillRouterEnabled() {
  return envBool("DYNAMO_PREFILL_ROUTER_ENABLED",
                 defaults::DYNAMO_PREFILL_ROUTER_ENABLED);
}

std::string dynamoPrefillRouterComponent() {
  return envString("DYNAMO_PREFILL_ROUTER_COMPONENT",
                   defaults::DYNAMO_PREFILL_ROUTER_COMPONENT);
}

std::string dynamoPrefillRouterEndpoint() {
  return envString("DYNAMO_PREFILL_ROUTER_ENDPOINT",
                   defaults::DYNAMO_PREFILL_ROUTER_ENDPOINT);
}

std::string dynamoPrefillRouterFallback() {
  return envString("DYNAMO_PREFILL_ROUTER_FALLBACK",
                   defaults::DYNAMO_PREFILL_ROUTER_FALLBACK);
}

/**
 * Mooncake KV Migration configuration.
 */
unsigned kvMigrationTimeoutMs() {
  return static_cast<unsigned>(
      envUlong("KV_MIGRATION_TIMEOUT_MS", defaults::KV_MIGRATION_TIMEOUT_MS));
}

unsigned kvMigrationSweepIntervalMs() {
  return static_cast<unsigned>(
      envUlong("KV_MIGRATION_SWEEP_INTERVAL_MS",
               defaults::KV_MIGRATION_SWEEP_INTERVAL_MS));
}

unsigned kvMigrationDrainPollMs() {
  return static_cast<unsigned>(envUlong("KV_MIGRATION_DRAIN_POLL_MS",
                                        defaults::KV_MIGRATION_DRAIN_POLL_MS));
}

}  // namespace tt::config
