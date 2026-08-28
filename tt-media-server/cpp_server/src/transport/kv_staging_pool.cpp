// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "transport/kv_staging_pool.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>

#include "utils/logger.hpp"

namespace tt::transport {

namespace {
// Read an unsigned tunable from the environment, falling back to `fallback` on
// unset/empty/unparseable/zero.
uint64_t envU64(const char* key, uint64_t fallback) {
  const char* v = std::getenv(key);
  if (v == nullptr || *v == '\0') return fallback;
  try {
    const unsigned long long parsed = std::stoull(v);
    return parsed == 0 ? fallback : static_cast<uint64_t>(parsed);
  } catch (...) {
    return fallback;
  }
}
}  // namespace

uint64_t defaultStagingBytes() {
  return envU64("TT_MOONCAKE_STAGING_BYTES", K_DEFAULT_STAGING_BYTES);
}

uint32_t stagingWindowDivisor() {
  return static_cast<uint32_t>(std::max<uint64_t>(
      1, envU64("TT_MOONCAKE_WINDOW_DIVISOR", K_DEFAULT_WINDOW_DIVISOR)));
}

KvStagingPool::KvStagingPool(std::shared_ptr<ITransferEngine> engine,
                             uint64_t bufferBytes, DeviceMapFn deviceMap) {
  if (!engine) {
    TT_LOG_ERROR("[KvStagingPool] no engine; not registered");
    return;
  }
  // Each buffer is double-pinned: engine-registered (ibv_reg_mr) and, on the
  // DRISC path, NOC-mapped so device reads DMA straight into staging. RAII
  // (DoublePinnedBuffer) unregisters/frees on destruction — no manual loop.
  bool allOk = true;
  for (auto& b : buffers_) {
    b = std::make_unique<DoublePinnedBuffer>(engine, bufferBytes, deviceMap);
    if (!b->registered()) {
      allOk = false;
      break;  // partials cleaned up by RAII; registered_ stays false
    }
  }
  registered_ = allOk;
  if (registered_) {
    TT_LOG_INFO(
        "[KvStagingPool] registered {} double-pinned buffers x {} bytes{}",
        kBuffers, bufferBytes, deviceMap ? " (NOC-mapped for DRISC)" : "");
  } else {
    TT_LOG_ERROR("[KvStagingPool] double-pinned buffer registration failed");
  }
}

KvStagingPool::~KvStagingPool() = default;  // DoublePinnedBuffer RAII

}  // namespace tt::transport
