// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <string>
#include <utility>

#include "domain/video_generate_request.hpp"
#include "runtime/dit/dit_types.hpp"

namespace tt::runtime {

/**
 * The server's video request as the runner's own type, field by field. The
 * server has already checked every field (prompt, step range, image_prompts
 * and references shape), so nothing is validated again here. The base64
 * images are copied once, into the runner's request.
 */
inline tt::dit::DiTRequest toDiTRequest(
    const tt::domain::VideoGenerateRequest& r) {
  tt::dit::DiTRequest out;
  out.prompt = r.prompt;
  out.negativePrompt = r.negativePrompt;
  out.numInferenceSteps = r.numInferenceSteps;
  out.seed = r.seed;
  if (r.fps) out.fps = static_cast<double>(*r.fps);
  out.height = r.height;
  out.width = r.width;
  out.numFrames = r.numFrames;
  if (r.duration) out.duration = static_cast<double>(*r.duration);
  out.aspectRatio = r.aspectRatio;
  if (r.imagePrompts) {
    out.imagePrompts.reserve(r.imagePrompts->size());
    for (const auto& entry : *r.imagePrompts) {
      tt::dit::ImagePrompt prompt;
      prompt.image = entry["image"].asString();
      prompt.framePos = entry.get("frame_pos", 0).asInt();
      out.imagePrompts.push_back(std::move(prompt));
    }
  }
  if (r.references) out.references = *r.references;
  out.taskId = std::to_string(r.task_id);
  return out;
}

}  // namespace tt::runtime
