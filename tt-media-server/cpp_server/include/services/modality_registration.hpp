// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

namespace tt::services {

/**
 * Populate ServiceRegistry / RunnerRegistry / RouteRegistry with the
 * built-in LLM and Embedding modalities. Idempotent. New modalities add
 * themselves next to the existing blocks.
 */
void registerBuiltinModalities();

}  // namespace tt::services
