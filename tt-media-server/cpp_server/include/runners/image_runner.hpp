// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/image_generate_request.hpp"

namespace tt::runners {

/**
 * Base class for image runners (SDXL today; DiT / Z-Image-Turbo to follow).
 *
 * Image generation is batch-1, single-shot, seconds long. Unlike LLM /
 * Embedding runners (which model long-running worker loops behind IRunner),
 * image runners are one-shot: warmup once, then process requests
 * synchronously. Concrete runners delegate the heavy lifting to Python
 * tt-metal pipelines via pybind11 while orchestrating LoRA state, prompt
 * prep, and post-processing entirely in C++.
 */
class ImageRunner {
 public:
  virtual ~ImageRunner() = default;

  /** Load weights, distribute to device, optionally do a warmup inference. */
  virtual bool warmup() = 0;

  /** Run inference for a single request and return the produced images as a
   * list of base64-encoded strings. Subclasses are expected to throw on
   * failure; the service translates exceptions to error responses. */
  virtual std::vector<std::string> run(
      const domain::ImageGenerateRequest& request) = 0;

  /** Release device resources / unload Python objects. */
  virtual void stop() {}

  /** Identifier for logs/metrics. */
  virtual const char* runnerType() const = 0;
};

/**
 * Construct an image runner from config. Throws on Python-side construction
 * failures (e.g. missing ttnn / diffusers); returns nullptr only for
 * genuinely unknown runner types.
 */
std::unique_ptr<ImageRunner> createImageRunner(const config::ImageConfig& cfg);

}  // namespace tt::runners
