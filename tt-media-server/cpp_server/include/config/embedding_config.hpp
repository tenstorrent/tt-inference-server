// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

#pragma once

namespace tt::config {

/**
 * Configuration for embedding service.
 * Currently a placeholder - will be expanded as embedding features are added.
 */
struct EmbeddingConfig {};

/**
 * Factory function to create EmbeddingConfig from environment variables and runtime settings.
 * Declared here, implemented in src/config/settings.cpp.
 */
EmbeddingConfig create_embedding_config();

}  // namespace tt::config
