// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

#include "config/constants.hpp"

#include <cstddef>
#include <string>

namespace tt::config {

/**
 * Central settings: use config/defaults when env is not set; env overrides when present.
 * Same model as tt-media-server/config/settings.py. All defaults live in constants.hpp defaults.
 */

/** Model service from MODEL_SERVICE. Default from defaults::MODEL_SERVICE. */
ModelService model_service();

/** True when model_service() == EMBEDDING. */
bool is_embedding_service();

/** True when model_service() == LLM. */
bool is_llm_service_enabled();

/** Number of worker processes = number of bracket pairs in DEVICE_IDS. */
size_t num_workers();

/** Max requests per batch (embedding). From MAX_BATCH_SIZE. Default: defaults::MAX_BATCH_SIZE. */
size_t batch_size();

/** Max wait (ms) to fill a batch. From MAX_BATCH_DELAY_TIME_MS. Default: defaults::MAX_BATCH_DELAY_TIME_MS. */
unsigned batch_timeout_ms();

/** Path prepended to Python sys.path for embedding runner. From TT_PYTHON_PATH. Default: defaults::TT_PYTHON_PATH. */
std::string python_path();

/** Runner type from MODEL_RUNNER. Default: defaults::MODEL_RUNNER. */
RunnerType runner_type();

/** Tokenizer path: tokenizers/tokenizer.json relative to executable. Empty if not found. */
std::string tokenizer_path();

/** Tokenizer config path: tokenizers/tokenizer_config.json relative to executable. Empty if not found. */
std::string tokenizer_config_path();

/**
 * Parse DEVICE_IDS and return the content inside the Nth bracket pair.
 * DEVICE_IDS format: "(0,1,2,3),(4,5,6,7)" → worker 0 gets "0,1,2,3", worker 1 gets "4,5,6,7".
 * This value is both the worker's identity and its TT_VISIBLE_DEVICES value,
 * matching the Python scheduler flow in model_services/scheduler.py.
 */
std::string visible_devices_for_worker(size_t worker_index);

}  // namespace tt::config
