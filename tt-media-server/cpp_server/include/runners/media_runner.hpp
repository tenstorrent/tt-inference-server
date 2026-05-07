// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

namespace tt::runners {

/**
 * Base for in-process media runners — owned by a service and dispatched to
 * synchronously, unlike the IPC-driven IRunner used by LLM/Embedding.
 */
template <typename Request, typename Response>
class MediaRunner {
 public:
  virtual ~MediaRunner() = default;

  virtual bool warmup() = 0;

  /** Throw on failure; the service maps exceptions to error responses. */
  virtual Response run(const Request& request) = 0;

  virtual void stop() {}

  virtual const char* runnerType() const = 0;
};

}  // namespace tt::runners
