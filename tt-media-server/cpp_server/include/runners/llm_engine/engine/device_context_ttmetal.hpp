// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#pragma once

#include "llm_engine/config.hpp"

namespace llm_engine {

/** Opens a 1x1 mesh device; sets config->mesh_device. Returns opaque context or nullptr. */
void* create_ttmetal_decode_context_and_config(Config* config);

/** Closes the mesh device and frees the context. No-op if ctx is nullptr. */
void destroy_ttmetal_decode_context(void* ctx);

}  // namespace llm_engine
