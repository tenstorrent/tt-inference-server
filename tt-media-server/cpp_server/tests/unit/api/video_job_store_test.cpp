// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#include "services/video_job_store.hpp"

#include <gtest/gtest.h>
#include <json/json.h>
#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "domain/video/video_job.hpp"
#include "domain/video/video_mode.hpp"
#include "domain/video_generate_request.hpp"
#include "utils/http_range.hpp"
#include "utils/video_faststart.hpp"

using tt::domain::VideoGenerateRequest;
using tt::domain::video::JobStatus;
using tt::domain::video::VideoJob;
using tt::services::JobStillActiveException;
using tt::services::MaxJobsReachedException;
using tt::services::VideoJobStore;

namespace {

std::filesystem::path tempClip(const std::string& name) {
  auto path =
      std::filesystem::temp_directory_path() /
      ("tt-video-job-test-" + std::to_string(::getpid()) + "-" + name + ".mp4");
  std::ofstream out(path, std::ios::trunc);
  out << "mp4";
  return path;
}

void removeQuietly(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

Json::Value params(const std::string& prompt) {
  Json::Value json;
  json["prompt"] = prompt;
  return json;
}

std::string statusOf(const Json::Value& job) {
  return job["status"].asString();
}

}  // namespace

TEST(VideoJobStoreTest, CreateGetAndListUseTaskIdOrder) {
  VideoJobStore store(10, 0);
  store.create(3, "tt-wan2.2", params("third"));
  store.create(1, "tt-wan2.2", params("first"));
  store.create(2, "tt-wan2.2", params("second"));

  auto first = store.get(1);
  ASSERT_TRUE(first.has_value());
  EXPECT_TRUE(tt::utils::isUuid((*first)["id"].asString()));
  EXPECT_EQ((*first)["job_type"].asString(), "video");
  EXPECT_EQ(statusOf(*first), "queued");
  EXPECT_EQ((*first)["model"].asString(), "tt-wan2.2");
  EXPECT_EQ((*first)["request_parameters"]["prompt"].asString(), "first");
  EXPECT_FALSE(first->isMember("completed_at"));
  EXPECT_FALSE(first->isMember("error"));
  EXPECT_FALSE(store.get(9).has_value());

  auto listed = store.list();
  ASSERT_EQ(listed.size(), 3u);
  // listed in task id order; each public id resolves back to its task id
  EXPECT_EQ(store.resolve(listed[0]["id"].asString()),
            std::optional<uint32_t>(1U));
  EXPECT_EQ(store.resolve(listed[1]["id"].asString()),
            std::optional<uint32_t>(2U));
  EXPECT_EQ(store.resolve(listed[2]["id"].asString()),
            std::optional<uint32_t>(3U));
}

TEST(VideoJobStoreTest, InProgressThenCompleteKeepsTheFileAndResultPath) {
  VideoJobStore store(10, 0);
  const auto clip = tempClip("complete");
  store.create(4, "tt-wan2.2", params("paper boat"));
  store.markInProgress(4);

  auto running = store.get(4);
  ASSERT_TRUE(running.has_value());
  EXPECT_EQ(statusOf(*running), "in_progress");

  store.complete(4, clip.string(), "");
  auto done = store.get(4);
  ASSERT_TRUE(done.has_value());
  EXPECT_EQ(statusOf(*done), "completed");
  EXPECT_TRUE(done->isMember("completed_at"));
  EXPECT_FALSE(done->isMember("error"));
  ASSERT_TRUE(store.resultPath(4).has_value());
  EXPECT_EQ(*store.resultPath(4), clip.string());
  EXPECT_TRUE(std::filesystem::is_regular_file(clip));
  removeQuietly(clip);
}

TEST(VideoJobStoreTest, MarkInProgressDoesNotOverrideCancel) {
  VideoJobStore store(10, 0);
  store.create(5, "tt-wan2.2", params("queued cancel"));
  ASSERT_TRUE(store.cancel(5).has_value());
  store.markInProgress(5);
  EXPECT_EQ(statusOf(*store.get(5)), "cancelled");

  store.create(6, "tt-wan2.2", params("running cancel"));
  store.markInProgress(6);
  ASSERT_TRUE(store.cancel(6).has_value());
  EXPECT_EQ(statusOf(*store.get(6)), "cancelling");
  store.markInProgress(6);
  EXPECT_EQ(statusOf(*store.get(6)), "cancelling");
}

TEST(VideoJobStoreTest, FailedCompletionKeepsTheFileOffTheResultPath) {
  VideoJobStore store(10, 0);
  const auto clip = tempClip("failed");
  store.create(7, "tt-wan2.2", params("fail"));
  store.markInProgress(7);
  store.complete(7, clip.string(), "device exploded");

  auto failed = store.get(7);
  ASSERT_TRUE(failed.has_value());
  EXPECT_EQ(statusOf(*failed), "failed");
  EXPECT_EQ((*failed)["error"]["code"].asString(), "processing_error");
  EXPECT_EQ((*failed)["error"]["message"].asString(), "device exploded");
  EXPECT_FALSE(store.resultPath(7).has_value());
  EXPECT_TRUE(std::filesystem::is_regular_file(clip));
  removeQuietly(clip);
}

TEST(VideoJobStoreTest, CancelQueuedIsTerminalAndDropsTheLaterFile) {
  VideoJobStore store(10, 0);
  const auto clip = tempClip("cancel-queued");
  store.create(8, "tt-wan2.2", params("cancel me"));
  auto cancelled = store.cancel(8);
  ASSERT_TRUE(cancelled.has_value());
  EXPECT_EQ(statusOf(*cancelled), "cancelled");
  EXPECT_TRUE(cancelled->isMember("completed_at"));
  EXPECT_FALSE(store.cancel(8).has_value());

  store.complete(8, clip.string(), "");
  EXPECT_EQ(statusOf(*store.get(8)), "cancelled");
  EXPECT_FALSE(store.resultPath(8).has_value());
  EXPECT_FALSE(std::filesystem::exists(clip));
}

TEST(VideoJobStoreTest, CancelRunningDropsTheFileWhenTheWorkerReturns) {
  VideoJobStore store(10, 0);
  const auto clip = tempClip("cancel-running");
  store.create(9, "tt-wan2.2", params("in flight"));
  store.markInProgress(9);
  auto cancelling = store.cancel(9);
  ASSERT_TRUE(cancelling.has_value());
  EXPECT_EQ(statusOf(*cancelling), "cancelling");
  EXPECT_FALSE(cancelling->isMember("completed_at"));
  EXPECT_FALSE(store.cancel(9).has_value());

  store.complete(9, clip.string(), "ignored because cancelled");
  EXPECT_EQ(statusOf(*store.get(9)), "cancelled");
  EXPECT_FALSE(store.get(9)->isMember("error"));
  EXPECT_FALSE(store.resultPath(9).has_value());
  EXPECT_FALSE(std::filesystem::exists(clip));
}

TEST(VideoJobStoreTest, UnknownAndTerminalCancelReturnEmpty) {
  VideoJobStore store(10, 0);
  EXPECT_FALSE(store.cancel(100).has_value());

  store.create(10, "tt-wan2.2", params("done"));
  store.complete(10, "mock://video/10", "");
  EXPECT_FALSE(store.cancel(10).has_value());

  store.create(11, "tt-wan2.2", params("bad"));
  store.complete(11, "", "nope");
  EXPECT_FALSE(store.cancel(11).has_value());
}

TEST(VideoJobStoreTest, DeleteRemovesOnlyTerminalJobsAndTheMp4) {
  VideoJobStore store(10, 0);
  const auto clip = tempClip("delete");
  store.create(12, "tt-wan2.2", params("keep until done"));
  EXPECT_THROW(store.remove(12), JobStillActiveException);
  EXPECT_TRUE(store.get(12).has_value());

  store.markInProgress(12);
  EXPECT_THROW(store.remove(12), JobStillActiveException);

  store.cancel(12);
  EXPECT_THROW(store.remove(12), JobStillActiveException);
  store.complete(12, clip.string(), "");
  EXPECT_FALSE(std::filesystem::exists(clip));

  const auto kept = tempClip("delete-kept");
  store.create(13, "tt-wan2.2", params("finished"));
  store.complete(13, kept.string(), "");
  EXPECT_TRUE(store.remove(13));
  EXPECT_FALSE(store.get(13).has_value());
  EXPECT_FALSE(store.resultPath(13).has_value());
  EXPECT_FALSE(std::filesystem::exists(kept));
  EXPECT_FALSE(store.remove(13));
  EXPECT_FALSE(store.remove(404));
}

TEST(VideoJobStoreTest, CompleteOfUnknownJobDeletesTheOrphanFile) {
  VideoJobStore store(10, 0);
  const auto clip = tempClip("orphan");
  store.complete(77, clip.string(), "");
  EXPECT_FALSE(store.get(77).has_value());
  EXPECT_FALSE(std::filesystem::exists(clip));
}

TEST(VideoJobStoreTest, MockPathIsNotTreatedAsAFile) {
  VideoJobStore store(10, 0);
  store.create(14, "tt-wan2.2", params("mock"));
  store.complete(14, "mock://video/14", "");
  ASSERT_TRUE(store.resultPath(14).has_value());
  EXPECT_EQ(*store.resultPath(14), "mock://video/14");
  EXPECT_TRUE(store.remove(14));
  EXPECT_FALSE(store.get(14).has_value());
}

TEST(VideoJobStoreTest, EmptyResultPathIsCompletedButNotDownloadable) {
  VideoJobStore store(10, 0);
  store.create(15, "tt-wan2.2", params("empty path"));
  store.complete(15, "", "");
  EXPECT_EQ(statusOf(*store.get(15)), "completed");
  EXPECT_FALSE(store.resultPath(15).has_value());
}

TEST(VideoJobStoreTest, EraseFreesTheSlotAndMaxJobsIsEnforced) {
  VideoJobStore store(2, 0);
  store.create(1, "tt-wan2.2", params("one"));
  store.create(2, "tt-wan2.2", params("two"));
  EXPECT_THROW(store.create(3, "tt-wan2.2", params("three")),
               MaxJobsReachedException);
  store.erase(2);
  EXPECT_FALSE(store.get(2).has_value());
  store.create(3, "tt-wan2.2", params("three"));
  EXPECT_TRUE(store.get(3).has_value());
  EXPECT_EQ(store.list().size(), 2u);
}

TEST(VideoJobStoreTest, ExpiredTerminalJobIsEvictedOnTheNextCreate) {
  VideoJobStore store(1, 1);
  const auto clip = tempClip("expired");
  store.create(1, "tt-wan2.2", params("old"));
  store.complete(1, clip.string(), "");
  EXPECT_THROW(store.create(2, "tt-wan2.2", params("blocked")),
               MaxJobsReachedException);

  std::this_thread::sleep_for(std::chrono::seconds(3));
  store.create(2, "tt-wan2.2", params("fresh"));
  EXPECT_FALSE(store.get(1).has_value());
  EXPECT_TRUE(store.get(2).has_value());
  EXPECT_FALSE(std::filesystem::exists(clip));
}

TEST(VideoJobStoreTest, QueuedJobIsNotEvictedByRetention) {
  VideoJobStore store(2, 1);
  store.create(1, "tt-wan2.2", params("still queued"));
  std::this_thread::sleep_for(std::chrono::seconds(3));
  store.create(2, "tt-wan2.2", params("also queued"));
  EXPECT_TRUE(store.get(1).has_value());
  EXPECT_EQ(statusOf(*store.get(1)), "queued");
}

TEST(VideoJobStoreTest, ConcurrentCreatesAreAllVisible) {
  constexpr int count = 32;
  VideoJobStore store(static_cast<size_t>(count), 0);
  std::vector<std::thread> threads;
  threads.reserve(count);
  for (int i = 1; i <= count; ++i) {
    threads.emplace_back([&store, i] {
      store.create(static_cast<uint32_t>(i), "tt-wan2.2", params("n"));
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(store.list().size(), static_cast<unsigned>(count));
  for (int i = 1; i <= count; ++i) {
    EXPECT_TRUE(store.get(static_cast<uint32_t>(i)).has_value());
  }
}

TEST(VideoJobJsonShapes, FailedJobAndExtraRequestParameters) {
  VideoJob failed;
  failed.taskId = 4;
  failed.model = "tt-wan2.2";
  failed.status = JobStatus::FAILED;
  failed.createdAt = 10;
  failed.completedAt = 12;
  failed.errorCode = "processing_error";
  failed.errorMessage = "device exploded";
  failed.requestParameters["prompt"] = "fail";

  const Json::Value one = failed.toPublicJson();
  EXPECT_EQ(one["status"].asString(), "failed");
  EXPECT_EQ(one["completed_at"].asInt64(), 12);
  EXPECT_EQ(one["error"]["code"].asString(), "processing_error");
  EXPECT_EQ(one["error"]["message"].asString(), "device exploded");
  EXPECT_TRUE(one["request_parameters"]["seed"].isNull());
  EXPECT_EQ(one["request_parameters"]["num_inference_steps"].asInt(), 20);

  // A key the Python schema does not have is kept, and a set value wins over
  // the default.
  VideoJob extra;
  extra.taskId = 5;
  extra.model = "tt-wan2.2";
  extra.createdAt = 1;
  extra.requestParameters["prompt"] = "boat";
  extra.requestParameters["num_inference_steps"] = 8;
  extra.requestParameters["guidance_scale"] = 5.0;
  const Json::Value json = extra.toPublicJson();
  EXPECT_EQ(json["request_parameters"]["num_inference_steps"].asInt(), 8);
  EXPECT_EQ(json["request_parameters"]["guidance_scale"].asDouble(), 5.0);
  EXPECT_TRUE(json["request_parameters"]["num_frames"].isNull());
  EXPECT_FALSE(json.isMember("completed_at"));
}

TEST(VideoModeTokens, EveryPythonTokenAndAGarbageValue) {
  using tt::domain::video::parseAsyncFlag;
  for (const char* token : {"1", "true", "TRUE", " yes ", "on"}) {
    auto parsed = parseAsyncFlag(token);
    ASSERT_TRUE(parsed.has_value()) << token;
    EXPECT_TRUE(*parsed) << token;
  }
  for (const char* token : {"0", "false", "No", "off", "\toff"}) {
    auto parsed = parseAsyncFlag(token);
    ASSERT_TRUE(parsed.has_value()) << token;
    EXPECT_FALSE(*parsed) << token;
  }
  EXPECT_FALSE(parseAsyncFlag("maybe").has_value());
  EXPECT_FALSE(parseAsyncFlag("").has_value());
}

TEST(VideoModeTokens, GarbageQueryOrBodyIsRejected) {
  ::unsetenv("USE_ASYNC_VIDEO");
  using tt::domain::video::videoRequestIsAsync;
  Json::Value body(Json::objectValue);
  EXPECT_THROW(videoRequestIsAsync("maybe", body), std::invalid_argument);
  EXPECT_THROW(videoRequestIsAsync("banana", body), std::invalid_argument);

  body["async"] = "banana";
  EXPECT_THROW(videoRequestIsAsync("", body), std::invalid_argument);
  body["async"] = 5;
  EXPECT_THROW(videoRequestIsAsync("", body), std::invalid_argument);
}

TEST(VideoModeTokens, QueryBeatsBodyBodyBeatsEnv) {
  ::unsetenv("USE_ASYNC_VIDEO");
  using tt::domain::video::videoRequestIsAsync;
  Json::Value body(Json::objectValue);
  body["async"] = "off";
  EXPECT_FALSE(videoRequestIsAsync("", body));
  EXPECT_TRUE(videoRequestIsAsync("true", body));

  body["async"] = Json::Value::null;
  EXPECT_TRUE(videoRequestIsAsync("", body));
  body["async"] = "yes";
  EXPECT_TRUE(videoRequestIsAsync("", body));

  ::setenv("USE_ASYNC_VIDEO", "", 1);
  body.removeMember("async");
  EXPECT_TRUE(videoRequestIsAsync("", body));
  ::setenv("USE_ASYNC_VIDEO", "false", 1);
  EXPECT_FALSE(videoRequestIsAsync("", body));
  ::unsetenv("USE_ASYNC_VIDEO");
}

TEST(VideoGenerateRequestGaps, RemainingTypesAndSparseJson) {
  auto reject = [](Json::Value json, const char* message) {
    try {
      VideoGenerateRequest::fromJson(json, 1);
      FAIL() << message;
    } catch (const std::invalid_argument& error) {
      EXPECT_EQ(std::string(error.what()), message);
    }
  };

  Json::Value json;
  json["prompt"] = "boat";
  json["negative_prompt"] = 1;
  reject(json, "negative_prompt must be a string");

  json = Json::Value(Json::objectValue);
  json["prompt"] = "boat";
  json["width"] = "832";
  reject(json, "width must be an integer");

  json = Json::Value(Json::objectValue);
  json["prompt"] = "boat";
  json["num_frames"] = 16.5;
  reject(json, "num_frames must be an integer");

  json = Json::Value(Json::objectValue);
  json["prompt"] = "boat";
  json["duration"] = "5";
  reject(json, "duration must be a number");

  json = Json::Value(Json::objectValue);
  json["prompt"] = "boat";
  json["aspect_ratio"] = 16;
  reject(json, "aspect_ratio must be a string");

  json = Json::Value(Json::objectValue);
  json["prompt"] = "boat";
  json["height"] = 1.5;
  reject(json, "height must be an integer");

  json = Json::Value(Json::objectValue);
  json["prompt"] = "boat";
  json["fps"] = 16;
  json["seed"] = -3;
  json["negative_prompt"] = "";
  json["aspect_ratio"] = " ";
  auto req = VideoGenerateRequest::fromJson(json, 1);
  ASSERT_TRUE(req.fps.has_value());
  EXPECT_FLOAT_EQ(*req.fps, 16.0F);
  ASSERT_TRUE(req.seed.has_value());
  EXPECT_EQ(*req.seed, -3);
  ASSERT_TRUE(req.negativePrompt.has_value());
  EXPECT_TRUE(req.negativePrompt->empty());
  ASSERT_TRUE(req.aspectRatio.has_value());
  EXPECT_EQ(*req.aspectRatio, " ");

  auto stored = req.toJson();
  EXPECT_EQ(stored["prompt"].asString(), "boat");
  EXPECT_EQ(stored["fps"].asFloat(), 16.0F);
  EXPECT_FALSE(stored.isMember("height"));
  EXPECT_EQ(stored["num_inference_steps"].asInt(), 20);  // defaulted
}

TEST(VideoFaststart, MissingInputFailsAndLeavesNoOutput) {
  const std::string out = "/tmp/tt-faststart-test-missing.mp4";
  std::filesystem::remove(out);
  EXPECT_FALSE(tt::utils::ensureFaststart("/tmp/definitely-not-here.mp4", out));
  EXPECT_EQ(tt::utils::faststartInPlace("/tmp/definitely-not-here.mp4"),
            tt::utils::FaststartResult::Failed);
}

TEST(VideoFaststart, InvalidFileIsLeftUntouched) {
  namespace fs = std::filesystem;
  const std::string path = "/tmp/tt-faststart-test-fake.mp4";
  {
    std::ofstream out(path, std::ios::binary);
    out << "not really an mp4";
  }
  EXPECT_FALSE(tt::utils::isFaststart(path).has_value());
  EXPECT_EQ(tt::utils::faststartInPlace(path),
            tt::utils::FaststartResult::Failed);
  std::ifstream in(path, std::ios::binary);
  std::string content((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
  EXPECT_EQ(content, "not really an mp4");
  EXPECT_FALSE(fs::exists(path + ".faststart.tmp.mp4"));
  fs::remove(path);
}

TEST(VideoFaststart, DetectsBoxOrderWithoutFfmpeg) {
  namespace fs = std::filesystem;
  auto box = [](const std::string& type, size_t payload) {
    std::string b(4, '\0');
    const uint32_t size = static_cast<uint32_t>(8 + payload);
    b[0] = static_cast<char>(size >> 24);
    b[1] = static_cast<char>(size >> 16);
    b[2] = static_cast<char>(size >> 8);
    b[3] = static_cast<char>(size);
    return b + type + std::string(payload, 'x');
  };
  const std::string moovFirst = "/tmp/tt-faststart-test-moovfirst.mp4";
  const std::string moovLast = "/tmp/tt-faststart-test-moovlast.mp4";
  {
    std::ofstream(moovFirst, std::ios::binary)
        << box("ftyp", 12) << box("moov", 40) << box("mdat", 100);
  }
  {
    std::ofstream(moovLast, std::ios::binary)
        << box("ftyp", 12) << box("free", 4) << box("mdat", 100)
        << box("moov", 40);
  }
  EXPECT_EQ(tt::utils::isFaststart(moovFirst), std::optional<bool>(true));
  EXPECT_EQ(tt::utils::isFaststart(moovLast), std::optional<bool>(false));
  // already faststart: ffmpeg is not needed, the file is not touched
  EXPECT_EQ(tt::utils::faststartInPlace(moovFirst),
            tt::utils::FaststartResult::AlreadyFaststart);
  fs::remove(moovFirst);
  fs::remove(moovLast);
}

TEST(VideoFaststart, RemuxesInPlaceWhenFfmpegExists) {
  namespace fs = std::filesystem;
  if (std::system("command -v ffmpeg >/dev/null 2>&1") != 0) {
    GTEST_SKIP() << "ffmpeg not installed";
  }
  const std::string path = "/tmp/tt-faststart-test-real.mp4";
  const std::string gen =
      "ffmpeg -y -loglevel error -f lavfi -i testsrc=size=64x64:rate=5 "
      "-t 1 -pix_fmt yuv420p " +
      path;
  ASSERT_EQ(std::system(gen.c_str()), 0);
  auto readAll = [&] {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
  };
  const std::string before = readAll();
  EXPECT_GT(before.find("moov"),
            before.find("mdat"));  // ffmpeg default: moov last
  EXPECT_EQ(tt::utils::isFaststart(path), std::optional<bool>(false));
  EXPECT_EQ(tt::utils::faststartInPlace(path),
            tt::utils::FaststartResult::Remuxed);
  const std::string after = readAll();
  ASSERT_EQ(after.substr(4, 4), "ftyp");
  EXPECT_LT(after.find("moov"), after.find("mdat"));  // now moov first
  EXPECT_EQ(tt::utils::isFaststart(path), std::optional<bool>(true));
  EXPECT_FALSE(fs::exists(path + ".faststart.tmp.mp4"));
  // a second call finds it already faststart and does not run ffmpeg again
  EXPECT_EQ(tt::utils::faststartInPlace(path),
            tt::utils::FaststartResult::AlreadyFaststart);
  fs::remove(path);
}

TEST(VideoOutputDir, InsideDirChecksResolvedPath) {
  namespace fs = std::filesystem;
  const fs::path root = "/tmp/tt-outdir-test";
  fs::remove_all(root);
  fs::create_directories(root);
  EXPECT_TRUE(tt::utils::isInsideDir((root / "a.mp4").string(), root));
  EXPECT_FALSE(tt::utils::isInsideDir(root.string(), root));
  EXPECT_FALSE(tt::utils::isInsideDir("/tmp/other.mp4", root));
  EXPECT_FALSE(tt::utils::isInsideDir((root / ".." / "x.mp4").string(), root));
  EXPECT_FALSE(tt::utils::isInsideDir("/tmp/tt-outdir-test2/a.mp4", root));
  fs::remove_all(root);
}

TEST(VideoOutputDir, StoreRefusesToDeleteOutsideRoot) {
  namespace fs = std::filesystem;
  const fs::path root = "/tmp/tt-outdir-store";
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path inside = root / "in.mp4";
  const fs::path outside = "/tmp/tt-outdir-outside.mp4";
  std::ofstream(inside) << "x";
  std::ofstream(outside) << "x";

  VideoJobStore store(10, 0);
  store.setOutputRoot(root);
  store.create(1, "m", Json::Value(Json::objectValue));
  store.create(2, "m", Json::Value(Json::objectValue));
  store.complete(1, inside.string(), "");
  store.complete(2, outside.string(), "");
  EXPECT_TRUE(store.remove(1));
  EXPECT_TRUE(store.remove(2));
  EXPECT_FALSE(fs::exists(inside));
  EXPECT_TRUE(fs::exists(outside));  // not deleted: outside the output dir
  fs::remove(outside);
  fs::remove_all(root);
}

TEST(VideoOutputDir, ResultOutsideTheRootFailsTheJob) {
  namespace fs = std::filesystem;
  const fs::path root = "/tmp/tt-outdir-complete";
  fs::remove_all(root);
  fs::create_directories(root);
  const fs::path inside = root / "in.mp4";
  const fs::path outside = "/tmp/tt-outdir-complete-outside.mp4";
  std::ofstream(inside) << "x";
  std::ofstream(outside) << "x";

  VideoJobStore store(10, 0);
  store.setOutputRoot(root);
  for (uint32_t id = 1; id <= 4; ++id) {
    store.create(id, "m", Json::Value(Json::objectValue));
  }
  store.complete(1, inside.string(), "");
  store.complete(2, outside.string(), "");
  store.complete(3, (root / ".." / "tt-outdir-complete-outside.mp4").string(),
                 "");
  store.complete(4, "mock://video/4", "");

  EXPECT_EQ((*store.get(1))["status"].asString(), "completed");
  EXPECT_TRUE(store.resultPath(1).has_value());

  for (uint32_t id : {2U, 3U}) {
    const auto job = *store.get(id);
    EXPECT_EQ(job["status"].asString(), "failed") << id;
    EXPECT_EQ(job["error"]["code"].asString(), "processing_error") << id;
    EXPECT_FALSE(store.resultPath(id).has_value()) << id;
  }
  EXPECT_TRUE(fs::exists(outside));  // a file outside the root is never touched

  EXPECT_EQ((*store.get(4))["status"].asString(), "completed");
  fs::remove(outside);
  fs::remove_all(root);
}

TEST(VideoOutputDir, PurgeRemovesOnlyTopLevelMp4) {
  namespace fs = std::filesystem;
  const fs::path root = "/tmp/tt-outdir-purge";
  fs::remove_all(root);
  fs::create_directories(root / "sub");
  std::ofstream(root / "a.mp4") << "x";
  std::ofstream(root / "b.txt") << "x";
  std::ofstream(root / "sub" / "c.mp4") << "x";
  EXPECT_EQ(tt::utils::purgeOrphanMp4(root), 1U);
  EXPECT_FALSE(fs::exists(root / "a.mp4"));
  EXPECT_TRUE(fs::exists(root / "b.txt"));
  EXPECT_TRUE(fs::exists(root / "sub" / "c.mp4"));
  EXPECT_EQ(tt::utils::purgeOrphanMp4("/tmp/tt-outdir-nope"), 0U);
  fs::remove_all(root);
}

TEST(VideoSweep, StuckInProgressJobFailsAndLateResultIsDropped) {
  namespace fs = std::filesystem;
  const fs::path file = "/tmp/tt-sweep-late.mp4";
  std::ofstream(file) << "x";
  VideoJobStore store(10, 0);
  store.create(1, "m", Json::Value(Json::objectValue));
  store.create(2, "m", Json::Value(Json::objectValue));
  store.markInProgress(1);
  // Job 2 is still queued: never counted as stuck.
  std::this_thread::sleep_for(std::chrono::seconds(2));
  auto r = store.sweep(1);
  EXPECT_EQ(r.stuck, 1U);
  auto j1 = store.get(1);
  EXPECT_EQ((*j1)["status"].asString(), "failed");
  EXPECT_EQ((*j1)["error"]["code"].asString(), "stale_job");
  EXPECT_EQ((*store.get(2))["status"].asString(), "queued");

  store.complete(1, file.string(), "");  // late result from the worker
  EXPECT_EQ((*store.get(1))["status"].asString(), "failed");
  EXPECT_FALSE(fs::exists(file));       // dropped, not served
  EXPECT_EQ(store.sweep(1).stuck, 0U);  // not stuck twice
}

TEST(VideoSweep, DisabledWhenZeroAndEvictsOldTerminalJobs) {
  VideoJobStore store(10, 1);  // retention 1 s
  store.create(1, "m", Json::Value(Json::objectValue));
  store.markInProgress(1);
  EXPECT_EQ(store.sweep(0).stuck, 0U);
  store.complete(1, "mock://video/1", "");
  std::this_thread::sleep_for(std::chrono::seconds(3));
  auto r = store.sweep(0);
  EXPECT_EQ(r.evicted, 1U);
  EXPECT_FALSE(store.get(1).has_value());
}

TEST(VideoJobStoreTest, CancelCompleteRaceEndsConsistently) {
  namespace fs = std::filesystem;
  constexpr int kJobs = 200;
  VideoJobStore store(kJobs + 10, 0);
  const fs::path dir = "/tmp/tt-race-test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  for (int i = 1; i <= kJobs; ++i) {
    store.create(i, "m", Json::Value(Json::objectValue));
    store.markInProgress(i);
    std::ofstream(dir / (std::to_string(i) + ".mp4")) << "x";
  }
  std::thread canceller([&] {
    for (int i = 1; i <= kJobs; ++i) store.cancel(i);
  });
  std::thread completer([&] {
    for (int i = kJobs; i >= 1; --i) {
      store.complete(i, (dir / (std::to_string(i) + ".mp4")).string(), "");
    }
  });
  canceller.join();
  completer.join();

  int cancelled = 0, completed = 0;
  for (int i = 1; i <= kJobs; ++i) {
    const std::string status = (*store.get(i))["status"].asString();
    const bool fileExists = fs::exists(dir / (std::to_string(i) + ".mp4"));
    if (status == "cancelled") {
      ++cancelled;
      EXPECT_FALSE(fileExists) << i;  // dropped with the cancelled result
      EXPECT_FALSE(store.resultPath(i).has_value()) << i;
    } else {
      ASSERT_EQ(status, "completed") << i;  // never stuck in cancelling
      ++completed;
      EXPECT_TRUE(fileExists) << i;
      EXPECT_TRUE(store.resultPath(i).has_value()) << i;
    }
  }
  EXPECT_EQ(cancelled + completed, kJobs);
  fs::remove_all(dir);
}

TEST(VideoJobId, PublicIdIsAUuidThatResolvesToTheTaskId) {
  VideoJobStore store(10, 0);
  auto json = store.create(7, "m", Json::Value(Json::objectValue));
  const std::string id = json["id"].asString();
  EXPECT_TRUE(tt::utils::isUuid(id));
  EXPECT_EQ(id[14], '4');  // version 4
  ASSERT_TRUE(store.resolve(id).has_value());
  EXPECT_EQ(*store.resolve(id), 7U);
  // upper case is accepted, unknown / malformed are not
  std::string upper = id;
  for (char& c : upper) c = static_cast<char>(std::toupper(c));
  EXPECT_EQ(store.resolve(upper), std::optional<uint32_t>(7U));
  EXPECT_FALSE(store.resolve("7").has_value());
  EXPECT_FALSE(
      store.resolve("00000000-0000-4000-8000-000000000000").has_value());
  EXPECT_EQ((*store.get(7))["id"].asString(), id);
}

TEST(VideoJobId, IdsAreUniqueAndForgottenWhenTheJobGoes) {
  VideoJobStore store(100, 0);
  std::set<std::string> ids;
  for (uint32_t i = 1; i <= 50; ++i) {
    ids.insert(
        store.create(i, "m", Json::Value(Json::objectValue))["id"].asString());
  }
  EXPECT_EQ(ids.size(), 50U);
  const std::string first = *ids.begin();
  const uint32_t task = *store.resolve(first);
  store.complete(task, "mock://video/x", "");
  EXPECT_TRUE(store.remove(task));
  EXPECT_FALSE(store.resolve(first).has_value());
  store.erase(*store.resolve(*std::next(ids.begin())));
  EXPECT_FALSE(store.resolve(*std::next(ids.begin())).has_value());
}

TEST(HttpRange, ParsesTheSingleRangeForms) {
  using tt::utils::ByteRange;
  using tt::utils::parseByteRange;
  auto r = parseByteRange("bytes=0-99", 1000);
  EXPECT_EQ(r.kind, ByteRange::Kind::Partial);
  EXPECT_EQ(r.offset, 0U);
  EXPECT_EQ(r.length, 100U);

  r = parseByteRange("bytes=900-", 1000);  // open ended
  EXPECT_EQ(r.offset, 900U);
  EXPECT_EQ(r.length, 100U);

  r = parseByteRange("bytes=-10", 1000);  // suffix
  EXPECT_EQ(r.offset, 990U);
  EXPECT_EQ(r.length, 10U);

  r = parseByteRange("bytes=-5000", 1000);  // suffix longer than the file
  EXPECT_EQ(r.offset, 0U);
  EXPECT_EQ(r.length, 1000U);

  r = parseByteRange("bytes=990-5000", 1000);  // end clamped
  EXPECT_EQ(r.offset, 990U);
  EXPECT_EQ(r.length, 10U);

  r = parseByteRange(" bytes = 5 - 5 ", 1000);  // spaces, one byte
  EXPECT_EQ(r.kind, ByteRange::Kind::Partial);
  EXPECT_EQ(r.offset, 5U);
  EXPECT_EQ(r.length, 1U);
}

TEST(HttpRange, UnsatisfiableIgnoredAndMultiRange) {
  using tt::utils::ByteRange;
  using tt::utils::parseByteRange;
  EXPECT_EQ(parseByteRange("bytes=1000-", 1000).kind,
            ByteRange::Kind::Unsatisfiable);
  EXPECT_EQ(parseByteRange("bytes=2000-3000", 1000).kind,
            ByteRange::Kind::Unsatisfiable);
  EXPECT_EQ(parseByteRange("bytes=-0", 1000).kind,
            ByteRange::Kind::Unsatisfiable);
  EXPECT_EQ(parseByteRange("bytes=0-9", 0).kind,
            ByteRange::Kind::Unsatisfiable);
  // ignored (whole file): no header, other unit, several ranges, bad syntax
  EXPECT_EQ(parseByteRange("", 1000).kind, ByteRange::Kind::None);
  EXPECT_EQ(parseByteRange("items=0-9", 1000).kind, ByteRange::Kind::None);
  EXPECT_EQ(parseByteRange("bytes=0-9,20-29", 1000).kind,
            ByteRange::Kind::None);
  EXPECT_EQ(parseByteRange("bytes=abc-9", 1000).kind, ByteRange::Kind::None);
  EXPECT_EQ(parseByteRange("bytes=9-3", 1000).kind, ByteRange::Kind::None);
  EXPECT_EQ(parseByteRange("bytes=5", 1000).kind, ByteRange::Kind::None);
}

// ---- Prepared for training (what Python's JobManager already does) ----------

using tt::services::ForceDeleteNotAllowedException;
using tt::services::JobTypePolicy;
namespace dv = tt::domain::video;

TEST(JobStoreTraining, JobTypeAndOrgAreInThePublicJson) {
  VideoJobStore store(10, 0);
  auto video = store.create(1, "m", Json::Value(Json::objectValue));
  EXPECT_EQ(video["job_type"].asString(), "video");
  EXPECT_FALSE(video.isMember("org_id"));
  auto training = store.create(2, "m", Json::Value(Json::objectValue),
                               dv::kJobTypeTraining, "org-a");
  EXPECT_EQ(training["job_type"].asString(), "training");
  EXPECT_EQ(training["org_id"].asString(), "org-a");
}

TEST(JobStoreTraining, OrgFilterHidesOtherOrganisations) {
  VideoJobStore store(10, 0);
  store.create(1, "m", Json::Value(Json::objectValue), dv::kJobTypeVideo, "a");
  store.create(2, "m", Json::Value(Json::objectValue), dv::kJobTypeVideo, "b");
  EXPECT_TRUE(store.get(1, "a").has_value());
  EXPECT_FALSE(store.get(1, "b").has_value());
  EXPECT_TRUE(store.get(1).has_value());  // no filter
  EXPECT_EQ(store.list("a").size(), 1U);
  EXPECT_EQ(store.list().size(), 2U);
  EXPECT_FALSE(store.cancel(2, "a").has_value());
  EXPECT_TRUE(store.cancel(2, "b").has_value());
  EXPECT_FALSE(store.remove(2, "a"));  // not visible: behaves as unknown
}

TEST(JobStoreTraining, WorkerIdentityIsKeptUntilTheJobCompletes) {
  VideoJobStore store(10, 0);
  store.create(1, "m", Json::Value(Json::objectValue));
  EXPECT_FALSE(store.assignedWorker(1).has_value());
  store.markInProgress(1, 2, 4242);
  ASSERT_TRUE(store.assignedWorker(1).has_value());
  EXPECT_EQ(store.assignedWorker(1)->workerIdx, 2);
  EXPECT_EQ(store.assignedWorker(1)->pid, 4242);
  store.complete(1, "mock://video/1", "");
  EXPECT_FALSE(store.assignedWorker(1).has_value());
}

TEST(JobStoreTraining, ReportProgressKeepsALongJobFromGoingStale) {
  VideoJobStore store(10, 0);
  store.create(1, "m", Json::Value(Json::objectValue), dv::kJobTypeTraining);
  store.markInProgress(1);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  store.reportProgress(1);  // a metric arrived
  EXPECT_EQ(store.sweep(1).stuck, 0U);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  EXPECT_EQ(store.sweep(1).stuck, 1U);  // silent for longer than the limit
}

TEST(JobStoreTraining, TrainingResultsAreNotAutoCleanedButVideoIs) {
  VideoJobStore store(10, 1);  // retention 1 s
  store.create(1, "m", Json::Value(Json::objectValue), dv::kJobTypeTraining);
  store.create(2, "m", Json::Value(Json::objectValue),
               dv::kJobTypeAdapterMerge);
  store.create(3, "m", Json::Value(Json::objectValue), dv::kJobTypeVideo);
  for (uint32_t id : {1U, 2U, 3U}) {
    store.markInProgress(id);
    store.complete(id, "mock://x", "");
  }
  std::this_thread::sleep_for(std::chrono::seconds(3));
  EXPECT_EQ(store.sweep(0).evicted,
            1U);  // only the video job (Python: exempt types)
  EXPECT_TRUE(store.get(1).has_value());
  EXPECT_TRUE(store.get(2).has_value());
  EXPECT_FALSE(store.get(3).has_value());
}

TEST(JobStoreTraining, DirectoryResultsAreRemovedOnlyInsideTheTypeRoot) {
  namespace fs = std::filesystem;
  const fs::path root = "/tmp/tt-train-root";
  const fs::path outside = "/tmp/tt-train-outside";
  fs::remove_all(root);
  fs::remove_all(outside);
  fs::create_directories(root / "adapter-1" / "sub");
  fs::create_directories(outside / "adapter-2");
  std::ofstream(root / "adapter-1" / "sub" / "w.bin") << "x";
  std::ofstream(outside / "adapter-2" / "w.bin") << "x";

  VideoJobStore store(10, 0);
  store.setJobTypePolicy(dv::kJobTypeTraining,
                         JobTypePolicy{false, root, true, true});
  for (uint32_t id : {1U, 2U}) {
    store.create(id, "m", Json::Value(Json::objectValue), dv::kJobTypeTraining);
    store.markInProgress(id);
  }
  store.complete(1, (root / "adapter-1").string(), "");
  store.complete(2, (outside / "adapter-2").string(), "");
  EXPECT_TRUE(store.remove(1));
  EXPECT_TRUE(store.remove(2));
  EXPECT_FALSE(fs::exists(root / "adapter-1"));    // whole directory removed
  EXPECT_TRUE(fs::exists(outside / "adapter-2"));  // outside the root: kept
  fs::remove_all(root);
  fs::remove_all(outside);
}

TEST(JobStoreTraining, VideoNeverRemovesADirectory) {
  namespace fs = std::filesystem;
  const fs::path dir = "/tmp/tt-video-dir-result";
  fs::remove_all(dir);
  fs::create_directories(dir);
  std::ofstream(dir / "x") << "x";
  VideoJobStore store(10, 0);
  store.create(1, "m", Json::Value(Json::objectValue));
  store.markInProgress(1);
  store.complete(1, dir.string(), "");
  EXPECT_TRUE(store.remove(1));
  EXPECT_TRUE(
      fs::exists(dir / "x"));  // plain remove() cannot delete a non-empty dir
  fs::remove_all(dir);
}

TEST(JobStoreTraining, ForceDeleteOnlyForTypesThatAllowIt) {
  VideoJobStore store(10, 0);
  store.create(1, "m", Json::Value(Json::objectValue), dv::kJobTypeTraining);
  store.create(2, "m", Json::Value(Json::objectValue), dv::kJobTypeVideo);
  store.markInProgress(1, 0, 100);
  store.markInProgress(2, 0, 101);
  EXPECT_THROW(store.remove(1), JobStillActiveException);  // no force
  EXPECT_THROW(store.remove(2, "", true), ForceDeleteNotAllowedException);
  ASSERT_TRUE(store.assignedWorker(1).has_value());  // read it first
  EXPECT_TRUE(store.remove(1, "", true));            // training: allowed
  EXPECT_FALSE(store.get(1).has_value());
  EXPECT_TRUE(store.get(2).has_value());
}

TEST(JobStoreTraining, MetricsLogsAndCheckpointsAreKeptPerJob) {
  VideoJobStore store(10, 0);
  store.create(1, "m", Json::Value(Json::objectValue), dv::kJobTypeTraining);
  Json::Value m1;
  m1["loss"] = 0.5;
  Json::Value m2;
  m2["loss"] = 0.4;
  EXPECT_TRUE(store.appendMetric(1, m1));
  EXPECT_TRUE(store.appendMetric(1, m2));
  EXPECT_TRUE(store.appendLog(1, Json::Value("started")));
  EXPECT_TRUE(store.appendCheckpoint(1, Json::Value("step-100")));
  EXPECT_FALSE(store.appendMetric(99, m1));  // unknown job
  ASSERT_TRUE(store.metrics(1).has_value());
  EXPECT_EQ(store.metrics(1)->size(), 2U);
  EXPECT_EQ((*store.metrics(1))[1]["loss"].asDouble(), 0.4);
  EXPECT_EQ(store.logs(1)->size(), 1U);
  EXPECT_EQ((*store.checkpoints(1))[0].asString(), "step-100");
  EXPECT_FALSE(store.metrics(99).has_value());
  // they are not part of the public job JSON (separate routes in Python)
  EXPECT_FALSE(store.get(1)->isMember("metrics"));
}
