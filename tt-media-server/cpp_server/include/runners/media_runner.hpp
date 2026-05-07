// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

namespace tt::runners {

/**
 * Base class for in-process media runners (image today; audio, video, TTS
 * next). Runners of this kind are owned directly by their service, dispatched
 * to synchronously, and are not driven by the IPC worker loop that backs the
 * IRunner-based LLM/Embedding runners.
 *
 * Each modality picks its own Request/Response types and typically declares a
 * `using FooRunner = MediaRunner<FooRequest, FooResponse>;` alias.
 */
template <typename Request, typename Response>
class MediaRunner {
 public:
  virtual ~MediaRunner() = default;

  virtual bool warmup() = 0;

  /** Subclasses are expected to throw on failure; the service translates
   * exceptions to error responses. */
  virtual Response run(const Request& request) = 0;

  virtual void stop() {}

  virtual const char* runnerType() const = 0;
};

}  // namespace tt::runners
