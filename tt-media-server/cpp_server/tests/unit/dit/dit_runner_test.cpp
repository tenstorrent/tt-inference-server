// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

// Tests of the C++ runner layer (runtime/dit) that need neither a device nor
// Python: the models (mesh, device parameters, validation, pipeline calls),
// the mp4 export (needs ffmpeg on PATH) and the multihost lockstep.

#include <gtest/gtest.h>
#include <json/json.h>
#include <stdlib.h>

#include <csignal>
#include <filesystem>
#include <fstream>

#include "runtime/dit/device_params.hpp"
#include "runtime/dit/dit_model.hpp"
#include "runtime/dit/dit_models.hpp"
#include "runtime/dit/dit_multihost.hpp"
#include "runtime/dit/media_export.hpp"

using namespace tt::dit;
namespace fs = std::filesystem;

namespace {

std::unique_ptr<DiTModel> make(const std::string& runner,
                               const std::string& device = "t3k") {
  RunnerConfig c;
  c.modelRunner = runner;
  c.device = device;
  c.outputDir = "/tmp/dit-test-unused";
  return createModel(c);
}

DiTRequest request(const std::string& json) {
  Json::Value v;
  Json::CharReaderBuilder b;
  std::string err;
  std::istringstream in(json);
  EXPECT_TRUE(Json::parseFromStream(b, in, &v, &err)) << err;
  return DiTRequest::fromJson(v);
}

const char* kImg = "aGVsbG8=";  // base64 content is not decoded in C++

struct EnvGuard {
  std::string name;
  explicit EnvGuard(const char* n, const char* value = nullptr) : name(n) {
    if (value)
      ::setenv(n, value, 1);
    else
      ::unsetenv(n);
  }
  ~EnvGuard() { ::unsetenv(name.c_str()); }
};

template <class E>
void expectRefused(const std::function<void()>& fn, const std::string& part) {
  try {
    fn();
    FAIL() << "expected a refusal containing: " << part;
  } catch (const E& e) {
    EXPECT_NE(std::string(e.what()).find(part), std::string::npos) << e.what();
  }
}

}  // namespace

// -- registry and tables ----------------------------------------------------

TEST(DiTRegistry, ServesTheWanRunners) {
  const std::vector<std::string> expected = {"tt-wan2.2", "tt-wan2.2-i2v"};
  EXPECT_EQ(supportedRunners(), expected);
}

TEST(DiTRegistry, DumpsMeshTableForTheParityCheckAgainstPython) {
  // tests/video/python compares this file with the Python server's
  // ModelConfigs.
  Json::Value out;
  for (const auto& name : supportedRunners()) {
    RunnerConfig c;
    c.modelRunner = name;
    // Any device: find them by probing the known device names.
    for (const char* device :
         {"n150", "n300", "t3k", "galaxy", "bh-galaxy", "p150", "p150x4",
          "p150x8", "p300", "p300x2", "galaxy_quad"}) {
      c.device = device;
      try {
        const auto m = createModel(c);
        out[name][device].append(m->mesh().rows);
        out[name][device].append(m->mesh().cols);
      } catch (const std::invalid_argument&) {
      }
    }
    EXPECT_TRUE(out.isMember(name)) << name;
  }
  std::ofstream file("dit_meshes.json");
  file << out.toStyledString();
}

TEST(DiTRegistry, RefusesUnknownRunnerAndDevice) {
  expectRefused<std::invalid_argument>([] { make("tt-wan2.3"); },
                                       "has no model");
  expectRefused<std::invalid_argument>([] { make("tt-wan2.2", "galaxy_octa"); },
                                       "DEVICE");
  EXPECT_EQ(make("tt-wan2.2")->outputKind(), OutputKind::Video);
}

TEST(DiTRegistry, MeshEnvOverrideAndCheckpoint) {
  {
    EnvGuard guard("SP_MESH_4X32", "true");
    EXPECT_EQ(make("tt-wan2.2")->mesh(), (MeshShape{4, 32}));
  }
  EnvGuard dir("MODEL_WEIGHTS_DIR");
  EXPECT_EQ(make("tt-wan2.2")->checkpoint(),
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers");
  RunnerConfig c;
  c.modelRunner = "tt-wan2.2";
  c.device = "t3k";
  c.modelWeightsPath = "/w";
  EXPECT_EQ(createModel(c)->checkpoint(), "/w");
  const fs::path tmp = fs::temp_directory_path() / "dit-weights-test";
  fs::create_directories(tmp);
  std::ofstream(tmp / "x") << "1";
  ::setenv("MODEL_WEIGHTS_DIR", tmp.c_str(), 1);
  c.modelRunner = "tt-wan2.2";
  EXPECT_EQ(createModel(c)->checkpoint(), tmp.string());
  fs::remove_all(tmp);
}

TEST(DiTRegistry, TracedFollowsEnvAndModelDefault) {
  EnvGuard unset("TT_VIDEO_TRACED");
  // The Python server traces everywhere; here a T3K runs Wan2.2 untraced by
  // default because traced output is noise there.
  EXPECT_FALSE(make("tt-wan2.2", "t3k")->traced());
  EXPECT_FALSE(make("tt-wan2.2-i2v", "t3k")->traced());
  EXPECT_TRUE(make("tt-wan2.2", "galaxy")->traced());
  ::setenv("TT_VIDEO_TRACED", "1", 1);
  EXPECT_TRUE(make("tt-wan2.2", "t3k")->traced());  // an explicit setting wins
  ::setenv("TT_VIDEO_TRACED", "0", 1);
  EXPECT_FALSE(make("tt-wan2.2", "galaxy")->traced());
}

// -- Wan2.2 -----------------------------------------------------------------

TEST(DiTWan, CallMatchesThePythonServer) {
  EnvGuard unset("TT_VIDEO_TRACED");
  const auto model = make("tt-wan2.2");
  Json::Value call = model->buildCall(request(R"({"prompt":"a cat"})"));
  Json::Value expected;
  expected["prompts"].append("a cat");
  expected["num_inference_steps"] = 20;
  expected["guidance_scale"] = 4.0;
  expected["guidance_scale_2"] = 3.0;
  expected["seed"] = 0;
  expected["traced"] = false;  // Wan2.2 on a T3K is untraced by default
  EXPECT_EQ(call, expected);

  call = model->buildCall(request(
      R"({"prompt":"a","negative_prompt":"blurry","num_inference_steps":4,"seed":7})"));
  EXPECT_EQ(call["negative_prompts"][0].asString(), "blurry");
  EXPECT_EQ(call["num_inference_steps"].asInt(), 4);
  EXPECT_EQ(call["seed"].asInt(), 7);
  // An empty negative prompt is not passed.
  EXPECT_FALSE(
      model->buildCall(request(R"({"prompt":"a","negative_prompt":""})"))
          .isMember("negative_prompts"));
}

TEST(DiTWan, ResolutionAndDeviceParams) {
  const auto t3k = make("tt-wan2.2");
  const Json::Value o = t3k->adapterOptions({});
  EXPECT_EQ(o["height"].asInt(), 480);
  EXPECT_EQ(o["width"].asInt(), 832);
  EXPECT_EQ(make("tt-wan2.2", "galaxy")->adapterOptions({})["height"].asInt(),
            720);

  EXPECT_EQ(t3k->deviceParams({false}), DeviceParams{});
  EXPECT_TRUE(make("tt-wan2.2", "galaxy")->deviceParams({false}).ring);
  const DeviceParams bh = make("tt-wan2.2", "bh-galaxy")->deviceParams({true});
  EXPECT_TRUE(bh.ring && bh.relaxedInit);
  EXPECT_EQ(bh.traceRegionBytes, 125'000'000);
  EXPECT_EQ(bh.routerPayloadBytes, 8192);
  EXPECT_TRUE(make("tt-wan2.2", "p150x4")->deviceParams({true}).ring);
}

TEST(DiTWan, ImageConditionedModelsCheckAndForwardImagePrompts) {
  const auto model = make("tt-wan2.2-i2v");
  expectRefused<RequestError>(
      [&] { model->validate(request(R"({"prompt":"x"})")); }, "image_prompts");
  auto two = [&](int a, int b) {
    return request(std::string(R"({"prompt":"x","image_prompts":[{"image":")") +
                   kImg + R"(","frame_pos":)" + std::to_string(a) +
                   R"(},{"image":")" + kImg + R"(","frame_pos":)" +
                   std::to_string(b) + "}]}");
  };
  expectRefused<RequestError>([&] { model->validate(two(0, 0)); },
                              "duplicate frame_pos");
  expectRefused<RequestError>([&] { model->validate(two(0, 81)); },
                              "frame_pos must be in");
  model->validate(two(0, -1));
  const Json::Value call = model->buildCall(two(0, -1));
  EXPECT_EQ(call["image_prompts"][1]["frame_pos"].asInt(), -1);
  EXPECT_EQ(call["guidance_scale"].asDouble(), 4.0);
  // The warmup request carries an image.
  EXPECT_EQ(model->warmupRequest().imagePrompts.size(), 1u);
  EXPECT_TRUE(make("tt-wan2.2")->warmupRequest().imagePrompts.empty());
}

TEST(DiTRequestTest, RoundTripAndRefusals) {
  const DiTRequest r =
      request(std::string(R"({"prompt":"x","task_id":5,"unknown":1,)") +
              R"("image_prompts":[{"image":"a","frame_pos":-1}]})");
  EXPECT_EQ(r.taskId, "5");
  EXPECT_EQ(r.imagePrompts[0].framePos, -1);
  const DiTRequest again = DiTRequest::fromJson(r.toJson());
  EXPECT_EQ(again.toJson(), r.toJson());
  Json::Value none;
  none["seed"] = 1;
  expectRefused<RequestError>([&] { DiTRequest::fromJson(none); }, "prompt");
}

// -- export -----------------------------------------------------------------

namespace {

bool haveFfmpeg() { return ::system("ffmpeg -version > /dev/null 2>&1") == 0; }

std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), {});
}

struct TempDir {
  fs::path path = fs::temp_directory_path() / ("dit-export-" + uuid4());
  TempDir() { fs::create_directories(path); }
  ~TempDir() { fs::remove_all(path); }
};

}  // namespace

TEST(DiTExport, EncodeArgsMatchThePythonCommand) {
  const auto a = encodeArgs(832, 480, 16, 23, "ultrafast", "/o.mp4");
  const std::string joined = [&] {
    std::string s;
    for (const auto& x : a) s += x + " ";
    return s;
  }();
  EXPECT_NE(joined.find("-s 832x480 -pix_fmt rgb24 -r 16 -i -"),
            std::string::npos);
  EXPECT_NE(joined.find("-c:v libx264 -crf 23 -pix_fmt yuv420p -tune film "
                        "-profile:v high -level 4.2"),
            std::string::npos);
  EXPECT_NE(joined.find("-preset ultrafast -movflags +faststart /o.mp4"),
            std::string::npos);
  const auto lossless = encodeArgs(64, 64, 16, 0, "", "/o.mp4");
  EXPECT_NE(std::find(lossless.begin(), lossless.end(), "yuv444p"),
            lossless.end());
  EXPECT_EQ(std::find(lossless.begin(), lossless.end(), "-preset"),
            lossless.end());
}

TEST(DiTExport, WritesAFaststartMp4FromFloatFrames) {
  if (!haveFfmpeg()) GTEST_SKIP() << "ffmpeg not on PATH";
  TempDir dir;
  std::vector<float> frames(1 * 9 * 64 * 96 * 3);
  for (size_t i = 0; i < frames.size(); ++i) frames[i] = (i % 251) / 250.0f;
  ArrayView v;
  v.data = frames.data();
  v.shape = {1, 9, 64, 96, 3};
  v.dtype = DType::F32;
  const std::string path = exportMp4(v, 16, dir.path.string());
  EXPECT_EQ(fs::path(path).parent_path(), dir.path);
  const std::string data = slurp(path);
  ASSERT_NE(data.find("moov"), std::string::npos);
  EXPECT_LT(data.find("moov"), data.find("mdat"));
}

TEST(DiTExport, GreyAndRgbaFramesAreAccepted) {
  if (!haveFfmpeg()) GTEST_SKIP() << "ffmpeg not on PATH";
  TempDir dir;
  for (int channels : {1, 4}) {
    std::vector<uint8_t> frames(4 * 32 * 32 * channels, 100);
    ArrayView v{
        frames.data(), {4, 32, 32, static_cast<size_t>(channels)}, DType::U8};
    EXPECT_NO_THROW(exportMp4(v, 8, dir.path.string())) << channels;
  }
  ArrayView bad{nullptr, {4, 32, 32}, DType::U8};
  EXPECT_THROW(exportMp4(bad, 8, dir.path.string()), std::runtime_error);
}

namespace {

size_t openFds() {
  size_t n = 0;
  for (const auto& e : fs::directory_iterator("/proc/self/fd")) {
    (void)e;
    ++n;
  }
  return n;
}

size_t ffmpegErrFiles() {
  size_t n = 0;
  for (const auto& e : fs::directory_iterator("/tmp")) {
    if (e.path().filename().string().rfind("tt-ffmpeg-err-", 0) == 0) ++n;
  }
  return n;
}

}  // namespace

TEST(DiTExport, Uint8RgbGoesOutUnchangedAndMatchesTheConvertedPath) {
  if (!haveFfmpeg()) GTEST_SKIP() << "ffmpeg not on PATH";
  TempDir dir;
  const size_t n = 5, h = 32, w = 48;
  std::vector<uint8_t> bytes(n * h * w * 3);
  std::vector<double> doubles(bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<uint8_t>((i * 7) % 256);
    doubles[i] = bytes[i];  // max > 1, so the float path does not rescale
  }
  const std::string fast =
      exportMp4({bytes.data(), {n, h, w, 3}, DType::U8}, 8, dir.path.string());
  const std::string general = exportMp4(
      {doubles.data(), {n, h, w, 3}, DType::F64}, 8, dir.path.string());
  EXPECT_EQ(slurp(fast), slurp(general));
}

TEST(DiTExport, AFailedEncodeLeavesNoFileDescriptorTempFileOrOutput) {
  if (!haveFfmpeg()) GTEST_SKIP() << "ffmpeg not on PATH";
  TempDir dir;
  // ffmpeg refuses the preset at start-up and stops reading its input while
  // frames are still being written: the write fails with EPIPE (SIGPIPE must
  // not kill this process) and the failure is an error, not a hang.
  EnvGuard preset("TT_VIDEO_EXPORT_PRESET", "no-such-preset");
  std::vector<uint8_t> frames(40 * 128 * 128 * 3, 9);
  const size_t fds = openFds(), errs = ffmpegErrFiles();
  for (int i = 0; i < 3; ++i) {
    EXPECT_THROW(exportMp4({frames.data(), {40, 128, 128, 3}, DType::U8}, 8,
                           dir.path.string()),
                 std::runtime_error);
  }
  EXPECT_EQ(openFds(), fds) << "descriptors leaked by failed exports";
  EXPECT_EQ(ffmpegErrFiles(), errs) << "ffmpeg stderr temp files left behind";
  EXPECT_TRUE(fs::is_empty(dir.path)) << "partial output left behind";
}

TEST(DiTExport, ExportDoesNotChangeTheSigpipeDisposition) {
  if (!haveFfmpeg()) GTEST_SKIP() << "ffmpeg not on PATH";
  struct sigaction before{};
  ::sigaction(SIGPIPE, nullptr, &before);
  TempDir dir;
  std::vector<uint8_t> frames(2 * 32 * 32 * 3, 9);
  exportMp4({frames.data(), {2, 32, 32, 3}, DType::U8}, 8, dir.path.string());
  struct sigaction after{};
  ::sigaction(SIGPIPE, nullptr, &after);
  EXPECT_EQ(before.sa_handler, after.sa_handler);
}

TEST(DiTExport, WavHasTheRiffHeaderAndPcm) {
  TempDir dir;
  std::vector<float> audio = {0.0f, 1.0f, -1.0f,
                              2.0f};  // 2 channels x 2 samples
  ArrayView v{audio.data(), {2, 2}, DType::F32};
  const std::string path = (dir.path / "a.wav").string();
  writeWav(v, 16000, path);
  const std::string data = slurp(path);
  ASSERT_EQ(data.size(), 44u + 8u);
  EXPECT_EQ(data.substr(0, 4), "RIFF");
  EXPECT_EQ(data.substr(8, 8), "WAVEfmt ");
  const int16_t* pcm = reinterpret_cast<const int16_t*>(data.data() + 44);
  EXPECT_EQ(pcm[0], 0);       // channel 0, sample 0
  EXPECT_EQ(pcm[1], -32767);  // channel 1, sample 0 (-1.0, 2.0 clipped below)
  EXPECT_EQ(pcm[2], 32767);   // channel 0, sample 1
  EXPECT_EQ(pcm[3], 32767);   // channel 1, sample 1 clipped from 2.0
}

TEST(DiTExport, MuxesAnAacSoundtrackAndLeavesNoTempFiles) {
  if (!haveFfmpeg()) GTEST_SKIP() << "ffmpeg not on PATH";
  TempDir dir;
  std::vector<float> frames(8 * 64 * 96 * 3, 0.0f);
  std::vector<float> audio(2 * 16000, 0.0f);
  AudioVideo av;
  av.frames = {frames.data(), {8, 64, 96, 3}, DType::F32};
  av.audio = {audio.data(), {2, 16000}, DType::F32};
  av.fps = 16;
  const std::string path = exportMp4WithAudio(av, dir.path.string());
  int files = 0;
  for (const auto& e : fs::directory_iterator(dir.path)) {
    ++files;
    EXPECT_EQ(e.path().string(), path);
  }
  EXPECT_EQ(files, 1);
  EXPECT_NE(slurp(path).find("mp4a"), std::string::npos);  // an audio track
}

// -- multihost --------------------------------------------------------------

namespace {

struct FakeComm : Comm {
  std::vector<std::string> sent;
  std::vector<std::string> inbox;
  int barriers = 0;
  bool failBroadcast = false;
  void broadcast(const std::string& m) override {
    if (failBroadcast) throw std::runtime_error("mpi down");
    sent.push_back(m);
  }
  std::string receive() override {
    std::string m = inbox.front();
    inbox.erase(inbox.begin());
    return m;
  }
  void barrier() override { ++barriers; }
};

struct StubRunner : IDiTRunner {
  std::vector<std::string>* events;
  std::runtime_error* failure = nullptr;
  bool requestError = false;
  bool exportError = false;
  explicit StubRunner(std::vector<std::string>* e) : events(e) {}
  void validate(const DiTRequest& r) override {
    events->push_back("validate " + r.prompt);
    if (r.prompt == "invalid") throw RequestError("refused up front");
  }
  void openDevice() override { events->push_back("open"); }
  void warmup() override { events->push_back("warmup"); }
  DiTResult generate(const DiTRequest& r) override {
    events->push_back("generate " + r.prompt);
    if (requestError) throw RequestError("refused");
    if (exportError) throw ExportError("disk full");
    if (failure) throw *failure;
    DiTResult out;
    out.path = "/tmp/x.mp4";
    return out;
  }
  void infer(const DiTRequest& r) override {
    events->push_back("infer " + r.prompt);
    if (r.prompt == "bad") throw RequestError("refused");
  }
  void close() override { events->push_back("close"); }
};

DiTRequest prompt(const std::string& p) {
  DiTRequest r;
  r.prompt = p;
  r.seed = 1;
  return r;
}

}  // namespace

TEST(DiTMultihost, MessagesRoundTrip) {
  const DiTRequest r = decodeRun(encodeRun(prompt("p")));
  EXPECT_EQ(r.prompt, "p");
  EXPECT_EQ(*r.seed, 1);
  EXPECT_TRUE(isShutdown(encodeShutdown()));
  EXPECT_THROW(decodeRun("garbage"), std::invalid_argument);
}

TEST(DiTMultihost, LockstepBroadcastsBeforeRunningAndShutsFollowersDown) {
  std::vector<std::string> events;
  auto comm = std::make_shared<FakeComm>();
  LockstepRunner lockstep(std::make_unique<StubRunner>(&events), comm);
  lockstep.warmup();
  EXPECT_EQ(comm->barriers, 1);
  EXPECT_EQ(lockstep.generate(prompt("p")).path, "/tmp/x.mp4");
  ASSERT_EQ(comm->sent.size(), 1u);
  EXPECT_EQ(decodeRun(comm->sent[0]).prompt, "p");
  lockstep.close();
  EXPECT_TRUE(isShutdown(comm->sent.back()));
  EXPECT_EQ(events.back(), "close");
}

TEST(DiTMultihost, ARefusedRequestIsAnErrorAndOthersEndTheJob) {
  std::vector<std::string> events, fatal;
  auto comm = std::make_shared<FakeComm>();
  auto stub = std::make_unique<StubRunner>(&events);
  stub->requestError = true;
  LockstepRunner refused(std::move(stub), comm,
                         [&](const std::string& r) { fatal.push_back(r); });
  EXPECT_THROW(refused.generate(prompt("p")), RequestError);
  EXPECT_TRUE(fatal.empty());

  std::runtime_error hang("hang");
  auto failing = std::make_unique<StubRunner>(&events);
  failing->failure = &hang;
  LockstepRunner broken(std::move(failing), comm,
                        [&](const std::string& r) { fatal.push_back(r); });
  EXPECT_THROW(broken.generate(prompt("p")), std::runtime_error);
  ASSERT_EQ(fatal.size(), 1u);
  EXPECT_NE(fatal[0].find("run failed"), std::string::npos);

  auto down = std::make_shared<FakeComm>();
  down->failBroadcast = true;
  LockstepRunner nobody(std::make_unique<StubRunner>(&events), down,
                        [&](const std::string& r) { fatal.push_back(r); });
  EXPECT_THROW(nobody.generate(prompt("p")), std::runtime_error);
  EXPECT_NE(fatal.back().find("broadcast"), std::string::npos);
}

TEST(DiTMultihost, ARequestRank0RefusesIsNeverSentToTheOtherRanks) {
  std::vector<std::string> events, fatal;
  auto comm = std::make_shared<FakeComm>();
  LockstepRunner lockstep(std::make_unique<StubRunner>(&events), comm,
                          [&](const std::string& r) { fatal.push_back(r); });
  EXPECT_THROW(lockstep.generate(prompt("invalid")), RequestError);
  EXPECT_TRUE(comm->sent.empty());
  EXPECT_TRUE(fatal.empty());
  for (const auto& e : events)
    EXPECT_EQ(e.rfind("generate", 0), std::string::npos);
}

TEST(DiTMultihost, AnExportFailureOnRank0DoesNotEndTheJob) {
  std::vector<std::string> events, fatal;
  auto comm = std::make_shared<FakeComm>();
  auto stub = std::make_unique<StubRunner>(&events);
  stub->exportError = true;
  LockstepRunner lockstep(std::move(stub), comm,
                          [&](const std::string& r) { fatal.push_back(r); });
  EXPECT_THROW(lockstep.generate(prompt("p")), ExportError);
  EXPECT_EQ(comm->sent.size(), 1u);  // the other ranks did run the request
  EXPECT_TRUE(fatal.empty());        // and every rank is still in step
}

TEST(DiTMultihost, FollowerMirrorsRank0UntilShutdown) {
  std::vector<std::string> events;
  auto comm = std::make_shared<FakeComm>();
  comm->inbox = {encodeRun(prompt("a")), encodeRun(prompt("bad")),
                 encodeRun(prompt("c")), encodeShutdown()};
  Follower follower(1, std::make_unique<StubRunner>(&events), comm);
  EXPECT_EQ(follower.run(), 0);
  const std::vector<std::string> expected = {"open",      "warmup",  "infer a",
                                             "infer bad", "infer c", "close"};
  EXPECT_EQ(events, expected);
  EXPECT_EQ(comm->barriers, 1);
}

TEST(DiTMultihost, FollowerReportsAFatalError) {
  std::vector<std::string> events;
  auto comm = std::make_shared<FakeComm>();
  comm->inbox = {"garbage"};
  Follower follower(1, std::make_unique<StubRunner>(&events), comm);
  EXPECT_EQ(follower.run(), 1);
}
