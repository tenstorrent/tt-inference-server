// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>
#include <vector>

#include "domain/image_generate_request.hpp"

namespace tt::runners {

/**
 * Base class for image runners. Image generation is batch-1 / one-shot, so
 * unlike the IRunner-based LLM/Embedding workers these are owned in-process
 * by ImageService and dispatched to synchronously.
 */
class ImageRunner {
 public:
  virtual ~ImageRunner() = default;

  virtual bool warmup() = 0;

  /** Subclasses are expected to throw on failure; the service translates
   * exceptions to error responses. */
  virtual std::vector<std::string> run(
      const domain::ImageGenerateRequest& request) = 0;

  virtual void stop() {}

  virtual const char* runnerType() const = 0;
};

}  // namespace tt::runners
