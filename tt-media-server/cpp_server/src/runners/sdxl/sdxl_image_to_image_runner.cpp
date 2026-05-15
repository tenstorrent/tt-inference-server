// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runners/sdxl/sdxl_image_to_image_runner.hpp"

#include <pybind11/stl.h>

#include <string>

namespace tt::runners::sdxl {

namespace {

constexpr const char* SDXL_BASE_REPO =
    "stabilityai/stable-diffusion-xl-base-1.0";

}  // namespace

SDXLImageToImageRunner::SDXLImageToImageRunner(
    const config::ImageConfig& config)
    : SDXLBaseRunner(config) {}

py::object SDXLImageToImageRunner::loadDiffusersPipeline() {
  py::module_ diffusers = py::module_::import("diffusers");
  std::string repo = config_.model_weights_path.empty()
                         ? std::string(SDXL_BASE_REPO)
                         : config_.model_weights_path;
  py::dict kwargs;
  kwargs["torch_dtype"] = torch_module_.attr("float32");
  kwargs["use_safetensors"] = py::bool_(true);
  return diffusers.attr("StableDiffusionXLImg2ImgPipeline")
      .attr("from_pretrained")(repo, **kwargs);
}

void SDXLImageToImageRunner::distributeBlock() {
  py::module_ pipelineMod = py::module_::import(
      "models.demos.stable_diffusion_xl_base.tt.tt_sdxl_img2img_pipeline");
  py::object cfgClass = pipelineMod.attr("TtSDXLImg2ImgPipelineConfig");
  py::dict cfgKwargs;
  cfgKwargs["encoders_on_device"] = py::bool_(true);
  cfgKwargs["is_galaxy"] = py::bool_(config_.is_galaxy);
  cfgKwargs["num_inference_steps"] = 2;
  cfgKwargs["guidance_scale"] = 5.0F;
  cfgKwargs["use_cfg_parallel"] = py::bool_(is_tensor_parallel_);
  py::object cfg = cfgClass(**cfgKwargs);

  py::dict pipelineKwargs;
  pipelineKwargs["ttnn_device"] = ttnn_device_;
  pipelineKwargs["torch_pipeline"] = pipeline_;
  pipelineKwargs["pipeline_config"] = cfg;
  tt_sdxl_ = pipelineMod.attr("TtSDXLImg2ImgPipeline")(**pipelineKwargs);
}

void SDXLImageToImageRunner::prepareInputTensorsForIteration(
    py::object tensors) {
  py::list args;
  args.append(tensors[py::int_(0)]);
  args.append(tensors[py::int_(1)][py::int_(0)]);
  args.append(tensors[py::int_(2)][py::int_(0)]);
  tt_sdxl_.attr("prepare_input_tensors")(args);
}

py::object SDXLImageToImageRunner::preprocessImage(
    const std::string& base64Image) const {
  py::module_ base64Mod = py::module_::import("base64");
  py::module_ ioMod = py::module_::import("io");
  py::module_ pil = py::module_::import("PIL.Image");

  std::string clean = base64Image;
  if (clean.compare(0, 5, "data:") == 0) {
    auto comma = clean.find(',');
    if (comma != std::string::npos) clean = clean.substr(comma + 1);
  }
  py::bytes raw = base64Mod.attr("b64decode")(clean).cast<py::bytes>();
  py::object buf = ioMod.attr("BytesIO")(raw);
  py::object pilImg = pil.attr("open")(buf);
  py::object converted = pilImg.attr("convert")("RGB");
  converted = converted.attr("resize")(
      py::make_tuple(config_.image_width, config_.image_height),
      pil.attr("Resampling").attr("LANCZOS"));

  py::object processor =
      tt_sdxl_.attr("torch_pipeline").attr("image_processor");
  py::dict kwargs;
  kwargs["height"] = config_.image_height;
  kwargs["width"] = config_.image_width;
  kwargs["crops_coords"] = py::none();
  kwargs["resize_mode"] = py::str("default");
  py::object tensor = processor.attr("preprocess")(converted, **kwargs);
  tensor = tensor.attr("to")(py::arg("dtype") = torch_module_.attr("float32"));
  return torch_module_.attr("cat")(py::make_tuple(tensor), py::arg("dim") = 0);
}

py::object SDXLImageToImageRunner::stackImageBatch(
    const std::vector<domain::ImageGenerateRequest>& requests) const {
  const size_t pad =
      batch_size_ > requests.size() ? batch_size_ - requests.size() : 0;
  py::list rows;
  for (const auto& r : requests) {
    rows.append(preprocessImage(r.image.value_or("")));
  }
  if (pad > 0) {
    py::object padRow = preprocessImage(requests.front().image.value_or(""));
    for (size_t i = 0; i < pad; ++i) rows.append(padRow);
  }
  return torch_module_.attr("cat")(rows, py::arg("dim") = 0);
}

bool SDXLImageToImageRunner::areBatchCompatible(
    const domain::ImageGenerateRequest& a,
    const domain::ImageGenerateRequest& b) const {
  return SDXLBaseRunner::areBatchCompatible(a, b) && a.strength == b.strength;
}

py::object SDXLImageToImageRunner::generateInputTensors(
    const std::vector<domain::ImageGenerateRequest>& requests,
    py::object promptEmbeds, py::object addTextEmbeds) {
  py::object torchImage = stackImageBatch(requests);
  const auto& head = requests.front();
  py::dict kwargs;
  kwargs["torch_image"] = torchImage;
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

domain::ImageGenerateRequest SDXLImageToImageRunner::warmupRequest() const {
  domain::ImageGenerateRequest r(0);
  r.prompt = "Sunrise on a beach";
  r.negative_prompt = "low resolution";
  r.num_inference_steps = 2;
  r.strength = 0.99F;
  r.image = "R0lGODdhAQABAPAAAP///wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==";
  return r;
}

void SDXLImageToImageRunner::applyModeSpecificSettings(
    const domain::ImageGenerateRequest& request) {
  if (request.strength.has_value()) {
    tt_sdxl_.attr("set_strength")(*request.strength);
  }
}

}  // namespace tt::runners::sdxl
