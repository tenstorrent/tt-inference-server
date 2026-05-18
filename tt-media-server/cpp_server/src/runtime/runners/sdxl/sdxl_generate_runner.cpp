// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/sdxl/sdxl_generate_runner.hpp"

#include <pybind11/stl.h>

#include <string>

namespace tt::runners::sdxl {

namespace {

constexpr const char* SDXL_BASE_REPO =
    "stabilityai/stable-diffusion-xl-base-1.0";

}  // namespace

SDXLGenerateRunner::SDXLGenerateRunner(const config::ImageConfig& config)
    : SDXLBaseRunner(config) {}

py::object SDXLGenerateRunner::loadDiffusersPipeline() {
  py::module_ diffusers = py::module_::import("diffusers");
  std::string repo = config_.model_weights_path.empty()
                         ? std::string(SDXL_BASE_REPO)
                         : config_.model_weights_path;
  py::dict kwargs;
  kwargs["torch_dtype"] = torch_module_.attr("float32");
  kwargs["use_safetensors"] = py::bool_(true);
  return diffusers.attr("DiffusionPipeline")
      .attr("from_pretrained")(repo, **kwargs);
}

void SDXLGenerateRunner::distributeBlock() {
  py::module_ pipelineMod = py::module_::import(
      "models.demos.stable_diffusion_xl_base.tt.tt_sdxl_pipeline");
  py::object cfgClass = pipelineMod.attr("TtSDXLPipelineConfig");
  py::dict cfgKwargs;
  cfgKwargs["encoders_on_device"] = py::bool_(true);
  cfgKwargs["is_galaxy"] = py::bool_(config_.is_galaxy);
  cfgKwargs["num_inference_steps"] = 2;
  cfgKwargs["guidance_scale"] = 5.0F;
  cfgKwargs["use_cfg_parallel"] = py::bool_(is_tensor_parallel_);
  cfgKwargs["image_resolution"] =
      py::make_tuple(config_.image_width, config_.image_height);
  py::object cfg = cfgClass(**cfgKwargs);

  py::dict pipelineKwargs;
  pipelineKwargs["ttnn_device"] = ttnn_device_;
  pipelineKwargs["torch_pipeline"] = pipeline_;
  pipelineKwargs["pipeline_config"] = cfg;
  tt_sdxl_ = pipelineMod.attr("TtSDXLPipeline")(**pipelineKwargs);
}

void SDXLGenerateRunner::prepareInputTensorsForIteration(py::object tensors) {
  py::list args;
  args.append(tensors[py::int_(0)]);
  args.append(tensors[py::int_(1)][py::int_(0)]);
  args.append(tensors[py::int_(2)][py::int_(0)]);
  tt_sdxl_.attr("prepare_input_tensors")(args);
}

py::object SDXLGenerateRunner::generateInputTensors(
    const std::vector<domain::ImageGenerateRequest>& requests,
    py::object promptEmbeds, py::object addTextEmbeds) {
  const auto& head = requests.front();
  py::dict kwargs;
  kwargs["all_prompt_embeds_torch"] = promptEmbeds;
  kwargs["torch_add_text_embeds"] = addTextEmbeds;
  kwargs["start_latent_seed"] =
      head.seed.has_value() ? py::cast(*head.seed) : py::none();
  kwargs["timesteps"] =
      head.timesteps.has_value() ? py::cast(*head.timesteps) : py::none();
  kwargs["sigmas"] =
      head.sigmas.has_value() ? py::cast(*head.sigmas) : py::none();
  return tt_sdxl_.attr("generate_input_tensors")(**kwargs);
}

domain::ImageGenerateRequest SDXLGenerateRunner::warmupRequest() const {
  domain::ImageGenerateRequest r(0);
  r.prompt = "Sunrise on a beach";
  r.prompt_2 = "Mountains in the background";
  r.negative_prompt = "low resolution";
  r.negative_prompt_2 = "blurry";
  r.num_inference_steps = 1;
  r.guidance_rescale = 0.7F;
  return r;
}

}  // namespace tt::runners::sdxl
