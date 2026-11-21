// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <optional>
#include <string>

namespace tt::domain::video {

/** Which kind of video request the served model takes. Same split as Python
 *  `I2V_MODEL_RUNNERS` (open_ai_api/video.py). */
enum class VideoDeployment {
  Text,       // text-to-video (Wan2.2 T2V)
  ImageOnly,  // Wan2.2 I2V: image_prompts are required
};

inline std::string lowerCase(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return value;
}

inline VideoDeployment classifyVideoRunner(const std::string& runnerName) {
  return lowerCase(runnerName) == "tt-wan2.2-i2v" ? VideoDeployment::ImageOnly
                                                  : VideoDeployment::Text;
}

/** The served runner: `MODEL_RUNNER`, the same variable the Python server
 *  reads. Also set for the mock runner, so the rules can be tested without a
 *  device. */
inline VideoDeployment servedVideoDeployment() {
  const char* runner = std::getenv("MODEL_RUNNER");
  return classifyVideoRunner(runner != nullptr ? runner : "");
}

/** Which route a request came in on. */
enum class VideoRoute { Text, I2V };

/** nullopt = the route fits the deployment; else the 422 message. The texts
 *  are Python's (open_ai_api/video.py `reject_*` dependencies). */
inline std::optional<std::string> routeRefusal(VideoRoute route,
                                               VideoDeployment d) {
  if (route == VideoRoute::Text && d == VideoDeployment::ImageOnly) {
    return "This deployment requires image conditioning. Use POST "
           "/generations/i2v with at least one image_prompts entry, or "
           "POST /generations/i2v/upload to send the image as a file.";
  }
  return std::nullopt;
}

}  // namespace tt::domain::video
