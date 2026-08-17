// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include "runtime/runners/sdxl/sdxl_base_runner.hpp"

namespace tt::runners::sdxl {

class SDXLGenerateRunner : public SDXLBaseRunner {
 public:
  explicit SDXLGenerateRunner(const config::ImageConfig& config);
  const char* runnerType() const override { return "SDXLGenerateRunner"; }

 protected:
  py::object loadDiffusersPipeline() override;
  void distributeBlock() override;
  void prepareInputTensorsForIteration(py::object tensors) override;
  py::object generateInputTensors(
      const std::vector<domain::ImageGenerateRequest>& requests,
      py::object promptEmbeds, py::object addTextEmbeds) override;
  domain::ImageGenerateRequest warmupRequest() const override;
};

}  // namespace tt::runners::sdxl
