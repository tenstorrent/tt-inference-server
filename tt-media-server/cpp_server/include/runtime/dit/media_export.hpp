// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tt::dit {

/** A random version-4 UUID, the name of an output file. */
std::string uuid4();

enum class DType { U8, F32, F64 };

/** Read-only view of a contiguous array a pipeline returned (numpy). The
 *  owner keeps the memory alive while the view is used. */
struct ArrayView {
  const void* data = nullptr;
  std::vector<size_t> shape;
  DType dtype = DType::U8;
  size_t count() const;
};

/** Frames plus a soundtrack (MiniMax-H3). `frames` is (N, H, W, 3) for rgb24
 *  or planar (N, H*3/2, W) uint8 for yuv420p; `audio` is (channels, samples)
 *  in [-1, 1]. */
struct AudioVideo {
  ArrayView frames;
  ArrayView audio;
  int samplingRate = 16000;
  double fps = 16.0;
  std::string pixelFormat = "rgb24";
};

/** Encode frames to an H.264 mp4 (libx264, CRF 23, ultrafast, faststart) in
 *  `outputDir`. Frames are (N, H, W, C) or (1, N, H, W, C) with C in 1, 3 or
 *  4, uint8 or float (floats in [0, 1] are scaled). Returns the path.
 *  TT_VIDEO_EXPORT_CRF / TT_VIDEO_EXPORT_PRESET override the quality. Same
 *  encoding as the Python server's VideoManager. */
std::string exportMp4(const ArrayView& frames, double fps,
                      const std::string& outputDir);

/** Encode the frames, then mux the soundtrack in as AAC. */
std::string exportMp4WithAudio(const AudioVideo& av,
                               const std::string& outputDir);

/** The ffmpeg arguments (without the binary) of the rawvideo -> mp4 encode. */
std::vector<std::string> encodeArgs(int width, int height, double fps, int crf,
                                    const std::string& preset,
                                    const std::string& outputPath,
                                    const std::string& pixelFormat = "rgb24");

/** (channels, samples) float audio in [-1, 1] as a 16-bit PCM WAV file. */
void writeWav(const ArrayView& audio, int samplingRate,
              const std::string& path);

}  // namespace tt::dit
