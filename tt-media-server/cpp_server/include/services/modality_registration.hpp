// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

namespace tt::services {

/**
 * Register the LLM and Embedding modalities into ServiceRegistry,
 * RunnerRegistry, and RouteRegistry. Idempotent.
 *
 * Called once from `service_factory::initializeServices()` before any
 * registry is queried. New modalities (image, video, audio, tts, cnn,
 * training) add themselves here next to the LLM/Embedding blocks; nothing
 * else in the codebase needs to know about them.
 */
void registerBuiltinModalities();

}  // namespace tt::services
