// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "config/settings.hpp"

#include <cstdlib>
#include <string>

namespace tt::config {

namespace {

std::string env_string(const char* name, const std::string& default_value) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : default_value;
}

unsigned long env_ulong(const char* name, unsigned long default_value) {
    const char* v = std::getenv(name);
    if (!v || !*v) return default_value;
    try {
        return std::stoul(v);
    } catch (const std::exception&) {
        return default_value;
    }
}

}  // namespace

ModelService model_service() {
    return model_service_from_string(env_string("TT_MODEL_SERVICE", to_string(ModelService::LLM)));
}

bool is_embedding_service() {
    return model_service() == ModelService::EMBEDDING;
}

bool is_llm_service_enabled() {
    return model_service() == ModelService::LLM;
}

size_t num_workers() {
    return static_cast<size_t>(env_ulong("TT_NUM_WORKERS", 4));
}

size_t batch_size() {
    return static_cast<size_t>(env_ulong("TT_BATCH_SIZE", 1));
}

unsigned batch_timeout_ms() {
    return static_cast<unsigned>(env_ulong("TT_BATCH_TIMEOUT_MS", 5));
}

std::string python_path() {
    return env_string("TT_PYTHON_PATH", "..");
}

RunnerType runner_type() {
    return runner_type_from_string(env_string("TT_RUNNER_TYPE", to_string(RunnerType::LLM_TEST)));
}

namespace {

unsigned device_offset() {
    return static_cast<unsigned>(env_ulong("TT_DEVICE_OFFSET", 1));
}

}  // namespace

std::string visible_devices_for_worker(size_t worker_id) {
    return std::to_string(worker_id + device_offset());
}

int visible_device_index_for_worker(size_t worker_id) {
    return static_cast<int>(worker_id + device_offset());
}

std::string device_id_for_worker(size_t worker_id) {
    return std::to_string(worker_id);
}

std::string worker_id_for_worker(size_t worker_id) {
    return std::to_string(worker_id);
}

}  // namespace tt::config
