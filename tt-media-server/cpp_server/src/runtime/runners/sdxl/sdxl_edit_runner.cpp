// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/runners/sdxl/sdxl_edit_runner.hpp"

#include <pybind11/stl.h>

#include <string>

namespace tt::runners::sdxl {

namespace {

constexpr const char* SDXL_INPAINTING_REPO =
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1";

}  // namespace

SDXLEditRunner::SDXLEditRunner(const config::ImageConfig& config)
    : SDXLImageToImageRunner(config) {}

py::object SDXLEditRunner::loadDiffusersPipeline() {
  py::module_ diffusers = py::module_::import("diffusers");
  std::string repo = config_.model_weights_path.empty()
                         ? std::string(SDXL_INPAINTING_REPO)
                         : config_.model_weights_path;
  py::dict kwargs;
  kwargs["torch_dtype"] = torch_module_.attr("float32");
  kwargs["use_safetensors"] = py::bool_(true);
  return diffusers.attr("DiffusionPipeline")
      .attr("from_pretrained")(repo, **kwargs);
}

void SDXLEditRunner::distributeBlock() {
  py::module_ pipelineMod = py::module_::import(
      "models.demos.stable_diffusion_xl_base.tt.tt_sdxl_inpainting_pipeline");
  py::object cfgClass = pipelineMod.attr("TtSDXLInpaintingPipelineConfig");
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
  tt_sdxl_ = pipelineMod.attr("TtSDXLInpaintingPipeline")(**pipelineKwargs);
}

py::object SDXLEditRunner::preprocessMask(const std::string& base64Mask) const {
  py::module_ base64Mod = py::module_::import("base64");
  py::module_ ioMod = py::module_::import("io");
  py::module_ pil = py::module_::import("PIL.Image");

  std::string clean = base64Mask;
  if (clean.compare(0, 5, "data:") == 0) {
    auto comma = clean.find(',');
    if (comma != std::string::npos) clean = clean.substr(comma + 1);
  }
  py::bytes raw = base64Mod.attr("b64decode")(clean).cast<py::bytes>();
  py::object buf = ioMod.attr("BytesIO")(raw);
  py::object pilImg = pil.attr("open")(buf);
  py::object converted = pilImg.attr("convert")("L");
  converted = converted.attr("resize")(
      py::make_tuple(config_.image_width, config_.image_height),
      pil.attr("Resampling").attr("LANCZOS"));

  py::object maskProcessor =
      tt_sdxl_.attr("torch_pipeline").attr("mask_processor");
  py::dict kwargs;
  kwargs["height"] = config_.image_height;
  kwargs["width"] = config_.image_width;
  kwargs["crops_coords"] = py::none();
  kwargs["resize_mode"] = py::str("default");
  py::object tensor = maskProcessor.attr("preprocess")(converted, **kwargs);
  return torch_module_.attr("cat")(py::make_tuple(tensor), py::arg("dim") = 0);
}

void SDXLEditRunner::prepareInputTensorsForIteration(py::object tensors) {
  py::list args;
  args.append(tensors[py::int_(0)]);
  args.append(tensors[py::int_(1)]);
  args.append(tensors[py::int_(2)]);
  args.append(tensors[py::int_(3)][py::int_(0)]);
  args.append(tensors[py::int_(4)][py::int_(0)]);
  tt_sdxl_.attr("prepare_input_tensors")(args);
}

py::object SDXLEditRunner::stackMaskBatch(
    const std::vector<domain::ImageGenerateRequest>& requests) const {
  const size_t pad =
      batch_size_ > requests.size() ? batch_size_ - requests.size() : 0;
  py::list rows;
  for (const auto& r : requests) {
    rows.append(preprocessMask(r.mask.value_or("")));
  }
  if (pad > 0) {
    py::object padRow = preprocessMask(requests.front().mask.value_or(""));
    for (size_t i = 0; i < pad; ++i) rows.append(padRow);
  }
  return torch_module_.attr("cat")(rows, py::arg("dim") = 0);
}

py::object SDXLEditRunner::generateInputTensors(
    const std::vector<domain::ImageGenerateRequest>& requests,
    py::object promptEmbeds, py::object addTextEmbeds) {
  py::object image = stackImageBatch(requests);
  py::object mask = stackMaskBatch(requests);
  py::object cond = mask.attr("__lt__")(0.5F);
  py::object maskedImage = image.attr("__mul__")(cond);
  const auto& head = requests.front();
  py::dict kwargs;
  kwargs["torch_image"] = image;
  kwargs["torch_masked_image"] = maskedImage;
  kwargs["torch_mask"] = mask;
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

domain::ImageGenerateRequest SDXLEditRunner::warmupRequest() const {
  domain::ImageGenerateRequest r = SDXLImageToImageRunner::warmupRequest();
  r.mask = "R0lGODdhAQABAPAAAP///wAAACH5BAAAAAAALAAAAAABAAEAAAICRAEAOw==";
  return r;
}

}  // namespace tt::runners::sdxl
