// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

namespace tt::services {

/**
 * Populates ServiceRegistry / RunnerRegistry / RouteRegistry with the
 * built-in `ModelService` entries (LLM, Embedding). Idempotent; new
 * services add themselves next to the existing blocks.
 */
void registerBuiltinModelServices();

}  // namespace tt::services
