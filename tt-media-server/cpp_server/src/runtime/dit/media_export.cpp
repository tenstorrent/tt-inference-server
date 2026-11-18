// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "runtime/dit/media_export.hpp"

#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

#include "runtime/dit/dit_env.hpp"

namespace tt::dit {

namespace {

constexpr int kDefaultCrf = 23;
constexpr const char* kDefaultPreset = "ultrafast";
constexpr int kFfmpegTimeoutS = 600;
// ffmpeg that takes no input for this long is stuck.
constexpr int kFeedStallS = 120;

double elementAt(const ArrayView& a, size_t i) {
  switch (a.dtype) {
    case DType::U8:
      return static_cast<const uint8_t*>(a.data)[i];
    case DType::F32:
      return static_cast<const float*>(a.data)[i];
    case DType::F64:
      return static_cast<const double*>(a.data)[i];
  }
  return 0;
}

uint8_t toByte(double v) {
  // astype(uint8) after clip(0, 255) truncates toward zero.
  return static_cast<uint8_t>(std::clamp(v, 0.0, 255.0));
}

std::string findInPath(const std::string& name) {
  const char* path = std::getenv("PATH");
  std::stringstream ss(path ? path : "");
  std::string dir;
  while (std::getline(ss, dir, ':')) {
    if (dir.empty()) continue;
    const std::filesystem::path p = std::filesystem::path(dir) / name;
    std::error_code ec;
    if (std::filesystem::is_regular_file(p, ec) &&
        ::access(p.c_str(), X_OK) == 0)
      return p.string();
  }
  throw std::runtime_error("ffmpeg not found on PATH");
}

std::filesystem::path newOutputPath(const std::string& dir) {
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return std::filesystem::path(dir) / (uuid4() + ".mp4");
}

struct Fd {
  int fd = -1;
  explicit Fd(int value = -1) : fd(value) {}
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  ~Fd() { reset(); }
  void reset() {
    if (fd >= 0) ::close(fd);
    fd = -1;
  }
};

struct RemoveOnExit {
  const char* path;
  ~RemoveOnExit() { std::remove(path); }
};

/** Blocks SIGPIPE on this thread while ffmpeg is fed, so a write to an ffmpeg
 *  that has died fails with EPIPE instead of killing the server, and drops the
 *  SIGPIPE that write left pending. The process-wide disposition is untouched.
 */
class SigpipeGuard {
 public:
  SigpipeGuard() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &set, &previous_);
  }
  ~SigpipeGuard() {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGPIPE);
    const timespec none{0, 0};
    while (::sigtimedwait(&set, nullptr, &none) > 0) {
    }
    pthread_sigmask(SIG_SETMASK, &previous_, nullptr);
  }
  SigpipeGuard(const SigpipeGuard&) = delete;
  SigpipeGuard& operator=(const SigpipeGuard&) = delete;

 private:
  sigset_t previous_;
};

/** In the forked child: ffmpeg must not hold the server's sockets or the
 *  device's file descriptors. */
void closeFdsFrom(int first) {
#ifdef SYS_close_range
  if (::syscall(SYS_close_range, static_cast<unsigned>(first), ~0U, 0) == 0) {
    return;
  }
#endif
  long limit = ::sysconf(_SC_OPEN_MAX);
  if (limit < 0 || limit > 65536) limit = 65536;
  for (int fd = first; fd < limit; ++fd) ::close(fd);
}

std::string readFile(const char* path) {
  std::ifstream in(path);
  std::stringstream text;
  text << in.rdbuf();
  return text.str();
}

/** Run ffmpeg; `feed` writes the raw video to its stdin (or is empty when
 *  ffmpeg reads files only). stderr goes to a temp file, so a chatty ffmpeg
 *  cannot fill a pipe and stall. Throws with ffmpeg's message on failure; no
 *  descriptor or temp file outlives the call. */
void runFfmpeg(const std::vector<std::string>& args,
               const std::function<void(int)>& feed) {
  const std::string binary = findInPath("ffmpeg");
  char errPath[] = "/tmp/tt-ffmpeg-err-XXXXXX";
  Fd errFd(::mkostemp(errPath, O_CLOEXEC));
  if (errFd.fd < 0) {
    throw std::runtime_error("cannot start ffmpeg: " +
                             std::string(std::strerror(errno)));
  }
  const RemoveOnExit removeErr{errPath};
  int ends[2] = {-1, -1};
  if (::pipe2(ends, O_CLOEXEC) != 0) {
    throw std::runtime_error("cannot start ffmpeg: " +
                             std::string(std::strerror(errno)));
  }
  Fd readEnd(ends[0]), writeEnd(ends[1]);
  std::vector<std::string> full = {binary, "-hide_banner", "-loglevel",
                                   "error"};
  full.insert(full.end(), args.begin(), args.end());
  std::vector<char*> argv;
  for (auto& a : full) argv.push_back(a.data());
  argv.push_back(nullptr);

  const pid_t pid = ::fork();
  if (pid < 0) throw std::runtime_error("fork failed for ffmpeg");
  if (pid == 0) {
    ::dup2(readEnd.fd, STDIN_FILENO);
    ::dup2(errFd.fd, STDERR_FILENO);
    const int devNull = ::open("/dev/null", O_WRONLY);
    if (devNull >= 0) ::dup2(devNull, STDOUT_FILENO);
    closeFdsFrom(3);
    ::execv(binary.c_str(), argv.data());
    ::_exit(127);
  }
  readEnd.reset();
  errFd.reset();
  ::fcntl(writeEnd.fd, F_SETFL, ::fcntl(writeEnd.fd, F_GETFL) | O_NONBLOCK);

  std::string feedFailure;
  {
    SigpipeGuard guard;
    try {
      if (feed) feed(writeEnd.fd);
    } catch (const std::exception& e) {
      feedFailure = e.what();
    }
    writeEnd.reset();
  }
  if (!feedFailure.empty()) {
    // The output is incomplete either way: stop ffmpeg instead of waiting.
    ::kill(pid, SIGKILL);
    int ignored = 0;
    ::waitpid(pid, &ignored, 0);
    const std::string message = readFile(errPath);
    throw std::runtime_error(
        feedFailure + (message.empty() ? "" : " (ffmpeg: " + message + ")"));
  }

  int status = 0;
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(kFfmpegTimeoutS);
  bool exited = false;
  while (std::chrono::steady_clock::now() < deadline) {
    if (::waitpid(pid, &status, WNOHANG) == pid) {
      exited = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!exited) {
    ::kill(pid, SIGKILL);
    ::waitpid(pid, &status, 0);
    throw std::runtime_error("ffmpeg export timed out");
  }
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("ffmpeg failed: " + readFile(errPath));
  }
}

/** Writes to ffmpeg's non-blocking stdin; waits for room, but not forever. */
void writeAll(int fd, const void* data, size_t size) {
  const char* p = static_cast<const char*>(data);
  while (size > 0) {
    const ssize_t n = ::write(fd, p, size);
    if (n < 0) {
      if (errno == EINTR) continue;
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        pollfd waiting{fd, POLLOUT, 0};
        const int ready = ::poll(&waiting, 1, kFeedStallS * 1000);
        if (ready == 0) {
          throw std::runtime_error("ffmpeg stopped reading its input for " +
                                   std::to_string(kFeedStallS) + " s");
        }
        if (ready < 0 && errno != EINTR) break;
        continue;
      }
      throw std::runtime_error("writing frames to ffmpeg failed: " +
                               std::string(std::strerror(errno)));
    }
    p += n;
    size -= static_cast<size_t>(n);
  }
  if (size > 0) {
    throw std::runtime_error("waiting to write frames to ffmpeg failed: " +
                             std::string(std::strerror(errno)));
  }
}

struct FrameGeometry {
  size_t frames = 0, height = 0, width = 0, channels = 0;
  size_t offset = 0;  // elements to skip (a leading batch of 1)
};

FrameGeometry geometryOf(const ArrayView& v) {
  FrameGeometry g;
  size_t first = 0;
  if (v.shape.size() == 5) {
    first = 1;  // (batch, N, H, W, C): the first video
  } else if (v.shape.size() != 4) {
    throw std::runtime_error("unexpected frame dimensions: " +
                             std::to_string(v.shape.size()) + " axes");
  }
  g.frames = v.shape[first];
  g.height = v.shape[first + 1];
  g.width = v.shape[first + 2];
  g.channels = v.shape[first + 3];
  if (g.channels != 1 && g.channels != 3 && g.channels != 4) {
    throw std::runtime_error("frames have " + std::to_string(g.channels) +
                             " channels, expected 1, 3 or 4");
  }
  return g;
}

template <typename T>
void convertFrame(const T* src, size_t pixels, size_t channels, double scale,
                  uint8_t* out) {
  switch (channels) {
    case 3:
      for (size_t i = 0; i < pixels * 3; ++i) {
        out[i] = toByte(static_cast<double>(src[i]) * scale);
      }
      break;
    case 1:  // grey: repeat
      for (size_t p = 0; p < pixels; ++p) {
        const uint8_t value = toByte(static_cast<double>(src[p]) * scale);
        out[p * 3] = out[p * 3 + 1] = out[p * 3 + 2] = value;
      }
      break;
    default:  // RGBA: drop A
      for (size_t p = 0; p < pixels; ++p) {
        for (size_t c = 0; c < 3; ++c) {
          out[p * 3 + c] =
              toByte(static_cast<double>(src[p * channels + c]) * scale);
        }
      }
  }
}

template <typename T>
double maxOfRange(const T* data, size_t count) {
  double m = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double value = static_cast<double>(data[i]);
    if (m < value) m = value;
  }
  return m;
}

double maxOf(const ArrayView& v, size_t begin, size_t count) {
  switch (v.dtype) {
    case DType::U8:
      return 255.0;
    case DType::F32:
      return maxOfRange(static_cast<const float*>(v.data) + begin, count);
    case DType::F64:
      return maxOfRange(static_cast<const double*>(v.data) + begin, count);
  }
  return 0.0;
}

/** One frame as packed RGB bytes. Floats in [0, 1] are scaled by 255; the
 *  range is decided from `max` (of the frame or of the whole stack). The dtype
 *  is dispatched once per frame, not per sample. */
void frameToRgb(const ArrayView& v, const FrameGeometry& g, size_t index,
                double max, std::vector<uint8_t>& out) {
  const double scale = (v.dtype != DType::U8 && max <= 1.0) ? 255.0 : 1.0;
  const size_t pixels = g.height * g.width;
  const size_t base = g.offset + index * pixels * g.channels;
  out.resize(pixels * 3);
  switch (v.dtype) {
    case DType::U8:
      convertFrame(static_cast<const uint8_t*>(v.data) + base, pixels,
                   g.channels, scale, out.data());
      break;
    case DType::F32:
      convertFrame(static_cast<const float*>(v.data) + base, pixels, g.channels,
                   scale, out.data());
      break;
    case DType::F64:
      convertFrame(static_cast<const double*>(v.data) + base, pixels,
                   g.channels, scale, out.data());
      break;
  }
}

/** Sends frame `index` to ffmpeg. uint8 RGB is already ffmpeg's rawvideo
 *  layout and goes out without a copy. */
void writeFrame(int fd, const ArrayView& v, const FrameGeometry& g,
                size_t index, double max, std::vector<uint8_t>& scratch) {
  if (v.dtype == DType::U8 && g.channels == 3) {
    const size_t bytes = g.height * g.width * 3;
    writeAll(fd, static_cast<const uint8_t*>(v.data) + g.offset + index * bytes,
             bytes);
    return;
  }
  frameToRgb(v, g, index, max, scratch);
  writeAll(fd, scratch.data(), scratch.size());
}

std::pair<int, std::string> encodeSettings() {
  int crf = kDefaultCrf;
  if (const auto v = envString("TT_VIDEO_EXPORT_CRF"); v && !v->empty()) {
    crf = std::clamp(std::stoi(*v), 0, 51);
  }
  return {
      crf,
      trimmed(envString("TT_VIDEO_EXPORT_PRESET").value_or(kDefaultPreset))};
}

}  // namespace

std::string uuid4() {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  uint8_t b[16];
  for (int i = 0; i < 2; ++i) {
    const uint64_t r = rng();
    std::memcpy(b + 8 * i, &r, 8);
  }
  b[6] = (b[6] & 0x0F) | 0x40;
  b[8] = (b[8] & 0x3F) | 0x80;
  char out[37];
  std::snprintf(
      out, sizeof out,
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
      b[12], b[13], b[14], b[15]);
  return out;
}

size_t ArrayView::count() const {
  size_t n = 1;
  for (size_t d : shape) n *= d;
  return shape.empty() ? 0 : n;
}

std::vector<std::string> encodeArgs(int width, int height, double fps, int crf,
                                    const std::string& preset,
                                    const std::string& outputPath,
                                    const std::string& pixelFormat) {
  std::ostringstream rate;
  rate << fps;
  std::vector<std::string> a = {
      "-y",
      "-f",
      "rawvideo",
      "-vcodec",
      "rawvideo",
      "-s",
      std::to_string(width) + "x" + std::to_string(height),
      "-pix_fmt",
      pixelFormat,
      "-r",
      rate.str(),
      "-i",
      "-"};
  if (crf == 0) {
    for (const char* s :
         {"-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p"})
      a.push_back(s);
  } else {
    for (const char* s : {"-c:v", "libx264", "-crf"}) a.push_back(s);
    a.push_back(std::to_string(crf));
    for (const char* s : {"-pix_fmt", "yuv420p", "-tune", "film", "-profile:v",
                          "high", "-level", "4.2"})
      a.push_back(s);
  }
  if (!preset.empty()) {
    a.push_back("-preset");
    a.push_back(preset);
  }
  a.push_back("-movflags");
  a.push_back("+faststart");
  a.push_back(outputPath);
  return a;
}

std::string exportMp4(const ArrayView& frames, double fps,
                      const std::string& outputDir) {
  const FrameGeometry g = geometryOf(frames);
  const std::string out = newOutputPath(outputDir).string();
  const auto [crf, preset] = encodeSettings();
  const size_t frameElems = g.height * g.width * g.channels;
  try {
    runFfmpeg(encodeArgs(static_cast<int>(g.width), static_cast<int>(g.height),
                         fps, crf, preset, out),
              [&](int fd) {
                std::vector<uint8_t> scratch;
                for (size_t i = 0; i < g.frames; ++i) {
                  // Streamed frame by frame: the range of float frames is
                  // decided per frame, as in the Python server.
                  writeFrame(
                      fd, frames, g, i,
                      maxOf(frames, g.offset + i * frameElems, frameElems),
                      scratch);
                }
              });
  } catch (...) {
    std::error_code ec;
    std::filesystem::remove(out, ec);  // no half-written file without a job
    throw;
  }
  return out;
}

void writeWav(const ArrayView& audio, int samplingRate,
              const std::string& path) {
  size_t channels = 0, samples = 0;
  if (audio.shape.size() == 1) {
    channels = 1;
    samples = audio.shape[0];
  } else if (audio.shape.size() == 2) {
    channels = audio.shape[0];
    samples = audio.shape[1];
  } else if (audio.shape.size() == 3 && audio.shape[0] == 1) {
    channels = audio.shape[1];
    samples = audio.shape[2];
  } else {
    throw std::runtime_error("expected audio shaped (channels, samples)");
  }
  const uint32_t dataBytes = static_cast<uint32_t>(channels * samples * 2);
  std::ofstream out(path, std::ios::binary);
  auto put32 = [&](uint32_t v) { out.write(reinterpret_cast<char*>(&v), 4); };
  auto put16 = [&](uint16_t v) { out.write(reinterpret_cast<char*>(&v), 2); };
  out.write("RIFF", 4);
  put32(36 + dataBytes);
  out.write("WAVEfmt ", 8);
  put32(16);
  put16(1);
  put16(static_cast<uint16_t>(channels));
  put32(static_cast<uint32_t>(samplingRate));
  put32(static_cast<uint32_t>(samplingRate * channels * 2));
  put16(static_cast<uint16_t>(channels * 2));
  put16(16);
  out.write("data", 4);
  put32(dataBytes);
  for (size_t s = 0; s < samples; ++s) {
    for (size_t c = 0; c < channels; ++c) {
      const double v = std::clamp(elementAt(audio, c * samples + s), -1.0, 1.0);
      const int16_t pcm = static_cast<int16_t>(v * 32767.0);
      out.write(reinterpret_cast<const char*>(&pcm), 2);
    }
  }
  if (!out) throw std::runtime_error("cannot write " + path);
}

std::string exportMp4WithAudio(const AudioVideo& av,
                               const std::string& outputDir) {
  namespace fs = std::filesystem;
  const fs::path output = newOutputPath(outputDir);
  const fs::path silent =
      output.parent_path() / (output.stem().string() + "_silent.mp4");
  const fs::path wav = fs::path(output).replace_extension(".wav");
  const auto [crf, preset] = encodeSettings();
  auto cleanup = [&] {
    std::error_code ec;
    fs::remove(silent, ec);
    fs::remove(wav, ec);
  };
  try {
    if (av.pixelFormat == "yuv420p") {
      // Planar (N, H*3/2, W) is already ffmpeg's rawvideo layout.
      if (av.frames.shape.size() != 3 || av.frames.dtype != DType::U8) {
        throw std::runtime_error("yuv420p frames must be uint8 (N, H*3/2, W)");
      }
      const size_t n = av.frames.shape[0], planar = av.frames.shape[1],
                   w = av.frames.shape[2];
      runFfmpeg(
          encodeArgs(static_cast<int>(w), static_cast<int>(planar * 2 / 3),
                     av.fps, crf, preset, silent.string(), "yuv420p"),
          [&](int fd) { writeAll(fd, av.frames.data, n * planar * w); });
    } else {
      const FrameGeometry g = geometryOf(av.frames);
      // The whole stack decides the float range here (a dark frame must not be
      // brightened on its own).
      const size_t frameElems = g.height * g.width * g.channels;
      const double max = maxOf(av.frames, g.offset, g.frames * frameElems);
      runFfmpeg(
          encodeArgs(static_cast<int>(g.width), static_cast<int>(g.height),
                     av.fps, crf, preset, silent.string()),
          [&](int fd) {
            std::vector<uint8_t> scratch;
            for (size_t i = 0; i < g.frames; ++i) {
              writeFrame(fd, av.frames, g, i, max, scratch);
            }
          });
    }
    writeWav(av.audio, av.samplingRate, wav.string());
    runFfmpeg({"-y", "-i", silent.string(), "-i", wav.string(), "-c:v", "copy",
               "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags",
               "+faststart", output.string()},
              nullptr);
  } catch (...) {
    cleanup();
    std::error_code ec;
    fs::remove(output, ec);
    throw;
  }
  cleanup();
  return output.string();
}

}  // namespace tt::dit
