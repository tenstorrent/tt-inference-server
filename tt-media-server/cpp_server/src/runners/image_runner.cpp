// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/image_runner.hpp"

#include "runners/sdxl/sdxl_runner.hpp"

namespace tt::runners {

std::unique_ptr<ImageRunner> createImageRunner(const config::ImageConfig& cfg) {
  // Construction failures from the SDXL runners (e.g. missing tt-metal /
  // ttnn / diffusers) are propagated as exceptions; the ImageService
  // constructor catches them and turns them into a startup failure with
  // the underlying error message intact. Returning nullptr is reserved
  // for genuinely unknown runner types.
  switch (cfg.runner_type) {
    case config::ModelRunnerType::TT_SDXL_GENERATE:
      return std::make_unique<sdxl::SDXLGenerateRunner>(cfg);
    case config::ModelRunnerType::TT_SDXL_IMAGE_TO_IMAGE:
      return std::make_unique<sdxl::SDXLImageToImageRunner>(cfg);
    case config::ModelRunnerType::TT_SDXL_EDIT:
      return std::make_unique<sdxl::SDXLEditRunner>(cfg);
    default:
      return nullptr;
  }
}

}  // namespace tt::runners
