// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "config/constants.hpp"

#include <cstddef>
#include <string>

namespace tt::config {

/**
 * Central settings: defaults with environment overrides.
 * Mimics tt-media-server/config/settings.py (env overrides defaults).
 * Uses constants.hpp enums; this is the only place that reads env for server config.
 */

/** Model service from TT_MODEL_SERVICE. Default: LLM. */
ModelService model_service();

/** True when model_service() == EMBEDDING. */
bool is_embedding_service();

/** True when model_service() == LLM. */
bool is_llm_service_enabled();

/** Number of worker processes. Default: 4. Env: TT_NUM_WORKERS. */
size_t num_workers();

/** Max requests per batch (embedding). Default: 1. Env: TT_BATCH_SIZE. */
size_t batch_size();

/** Max wait (ms) to fill a batch. Default: 5. Env: TT_BATCH_TIMEOUT_MS. */
unsigned batch_timeout_ms();

/** Path prepended to Python sys.path for embedding runner. Default: "..". Env: TT_PYTHON_PATH. */
std::string python_path();

/** Runner type from TT_RUNNER_TYPE. Default: LLM_TEST. Env: TT_RUNNER_TYPE. */
RunnerType runner_type();

/**
 * Values to set in worker process env (used by embedding_service before creating runner).
 * TT_DEVICE_OFFSET (default 1): visible device index = worker_id + offset.
 */
std::string visible_devices_for_worker(size_t worker_id);
/** Visible device index (1-based by default) for runner ctor and logging. */
int visible_device_index_for_worker(size_t worker_id);
std::string device_id_for_worker(size_t worker_id);
std::string worker_id_for_worker(size_t worker_id);

}  // namespace tt::config
