// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <istream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "config/runner_config.hpp"
#include "domain/tts/tts_types.hpp"
#include "ipc/boost/boost_cancel_queue.hpp"
#include "ipc/boost/boost_memory_queue.hpp"
#include "ipc/interface/cancel_queue.hpp"
#include "ipc/serialization.hpp"

namespace tt::ipc::tts {

namespace ser = tt::ipc::serialization;

struct TtsQueueNames {
  std::string taskQueue = "tt_tts_task_queue";
  std::string audioQueuePrefix = "tt_tts_audio_queue_";
  std::string cancelQueuePrefix = "tt_tts_cancel_queue_";
};

struct TtsIpcTask {
  uint32_t task_id = 0;
  uint32_t flags = 0;
  std::string text;
  std::optional<std::string> description;
  std::vector<uint32_t> promptTokens;
  std::vector<int16_t> voiceWavPcm;
  domain::tts::TtsGenerationParams generation;

  static constexpr uint32_t FLAG_DONE = 1;

  bool isDone() const { return (flags & FLAG_DONE) != 0; }

  static TtsIpcTask done() {
    TtsIpcTask task;
    task.flags = FLAG_DONE;
    return task;
  }

  static TtsIpcTask fromDomainTask(const domain::tts::TtsTask& task) {
    TtsIpcTask ipcTask;
    ipcTask.task_id = task.task_id;
    ipcTask.text = task.text;
    ipcTask.description = task.description;
    ipcTask.promptTokens = task.promptTokens;
    ipcTask.voiceWavPcm = task.voiceWavPcm;
    ipcTask.generation = task.generation;
    return ipcTask;
  }

  domain::tts::TtsTask toDomainTask() const {
    domain::tts::TtsTask task;
    task.task_id = task_id;
    task.text = text;
    task.description = description;
    task.promptTokens = promptTokens;
    task.voiceWavPcm = voiceWavPcm;
    task.generation = generation;
    return task;
  }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&task_id), sizeof(task_id));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    ser::writeString(os, text);
    const bool hasDescription = description.has_value();
    os.write(reinterpret_cast<const char*>(&hasDescription),
             sizeof(hasDescription));
    if (hasDescription) {
      ser::writeString(os, *description);
    }
    ser::writeVector(os, promptTokens);
    ser::writeVector(os, voiceWavPcm);
    os.write(reinterpret_cast<const char*>(&generation.ignoreEos),
             sizeof(generation.ignoreEos));
    ser::writeVector(os, generation.stopTokenIds);
  }

  static TtsIpcTask deserialize(std::istream& is) {
    TtsIpcTask task;
    is.read(reinterpret_cast<char*>(&task.task_id), sizeof(task.task_id));
    is.read(reinterpret_cast<char*>(&task.flags), sizeof(task.flags));
    task.text = ser::readString(is);
    bool hasDescription = false;
    is.read(reinterpret_cast<char*>(&hasDescription), sizeof(hasDescription));
    if (hasDescription) {
      task.description = ser::readString(is);
    }
    task.promptTokens = ser::readVector<uint32_t>(is);
    task.voiceWavPcm = ser::readVector<int16_t>(is);
    is.read(reinterpret_cast<char*>(&task.generation.ignoreEos),
            sizeof(task.generation.ignoreEos));
    task.generation.stopTokenIds = ser::readVector<uint32_t>(is);
    return task;
  }
};

struct TtsAudioChunkMessage {
  uint32_t task_id = 0;
  uint32_t chunkIndex = 0;
  uint32_t flags = 0;
  uint32_t sampleRateHz = 0;
  uint16_t channels = 0;
  std::vector<uint16_t> samplesBf16;
  std::string error;

  static constexpr uint32_t FLAG_FINAL = 1;
  static constexpr uint32_t FLAG_ERROR = 2;
  static constexpr uint32_t FLAG_DONE = 4;
  static constexpr uint32_t FLAG_CANCELLED = 8;

  bool isFinal() const { return (flags & FLAG_FINAL) != 0; }
  bool isError() const { return (flags & FLAG_ERROR) != 0; }
  bool isDone() const { return (flags & FLAG_DONE) != 0; }
  bool isCancelled() const { return (flags & FLAG_CANCELLED) != 0; }

  static TtsAudioChunkMessage fromDomainChunk(
      uint32_t taskId, const domain::tts::TtsAudioChunk& chunk) {
    TtsAudioChunkMessage message;
    message.task_id = taskId;
    message.chunkIndex = chunk.chunkIndex;
    message.sampleRateHz = chunk.sampleRateHz;
    message.channels = chunk.channels;
    message.samplesBf16 = chunk.samplesBf16;
    return message;
  }

  static TtsAudioChunkMessage finish(
      uint32_t taskId,
      domain::tts::TtsFinishReason reason =
          domain::tts::TtsFinishReason::Completed,
      std::string error = {}) {
    TtsAudioChunkMessage message;
    message.task_id = taskId;
    message.flags = FLAG_FINAL;
    message.error = std::move(error);
    switch (reason) {
      case domain::tts::TtsFinishReason::Completed:
        break;
      case domain::tts::TtsFinishReason::Cancelled:
        message.flags |= FLAG_CANCELLED;
        break;
      case domain::tts::TtsFinishReason::Error:
        message.flags |= FLAG_ERROR;
        break;
    }
    return message;
  }

  static TtsAudioChunkMessage done() {
    TtsAudioChunkMessage message;
    message.flags = FLAG_DONE;
    return message;
  }

  domain::tts::TtsFinishReason finishReason() const {
    if (isCancelled()) {
      return domain::tts::TtsFinishReason::Cancelled;
    }
    if (isError()) {
      return domain::tts::TtsFinishReason::Error;
    }
    return domain::tts::TtsFinishReason::Completed;
  }

  domain::tts::TtsAudioChunk toDomainChunk() const {
    domain::tts::TtsAudioChunk chunk;
    chunk.task_id = task_id;
    chunk.chunkIndex = chunkIndex;
    chunk.samplesBf16 = samplesBf16;
    chunk.sampleRateHz = sampleRateHz;
    chunk.channels = channels;
    return chunk;
  }

  void serialize(std::ostream& os) const {
    os.write(reinterpret_cast<const char*>(&task_id), sizeof(task_id));
    os.write(reinterpret_cast<const char*>(&chunkIndex), sizeof(chunkIndex));
    os.write(reinterpret_cast<const char*>(&flags), sizeof(flags));
    os.write(reinterpret_cast<const char*>(&sampleRateHz),
             sizeof(sampleRateHz));
    os.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    ser::writeVector(os, samplesBf16);
    ser::writeString(os, error);
  }

  static TtsAudioChunkMessage deserialize(std::istream& is) {
    TtsAudioChunkMessage message;
    is.read(reinterpret_cast<char*>(&message.task_id), sizeof(message.task_id));
    is.read(reinterpret_cast<char*>(&message.chunkIndex),
            sizeof(message.chunkIndex));
    is.read(reinterpret_cast<char*>(&message.flags), sizeof(message.flags));
    is.read(reinterpret_cast<char*>(&message.sampleRateHz),
            sizeof(message.sampleRateHz));
    is.read(reinterpret_cast<char*>(&message.channels),
            sizeof(message.channels));
    message.samplesBf16 = ser::readVector<uint16_t>(is);
    message.error = ser::readString(is);
    return message;
  }
};

class TtsTaskQueue {
 public:
  using Queue = boost::MemoryQueue<TtsIpcTask, 2 * 1024 * 1024>;

  TtsTaskQueue(const std::string& name, int capacity)
      : queue(std::make_unique<Queue>(name, capacity)) {}

  explicit TtsTaskQueue(const std::string& name)
      : queue(Queue::openExisting(name)) {}

  void push(const TtsIpcTask& task) { queue->push(task); }
  bool tryPush(const TtsIpcTask& task) { return queue->tryPush(task); }
  bool tryPop(TtsIpcTask& out) { return queue->tryPop(out); }
  void receive(TtsIpcTask& out) { queue->receive(out); }
  bool empty() const { return queue->empty(); }
  void shutdown() { queue->push(TtsIpcTask::done()); }
  void remove() { queue->remove(); }

 private:
  std::unique_ptr<Queue> queue;
};

class TtsAudioChunkQueue {
 public:
  using Queue = boost::MemoryQueue<TtsAudioChunkMessage, 512 * 1024>;

  TtsAudioChunkQueue(const std::string& name, int capacity)
      : queue(std::make_unique<Queue>(name, capacity)) {}

  explicit TtsAudioChunkQueue(const std::string& name)
      : queue(Queue::openExisting(name)) {}

  bool push(const TtsAudioChunkMessage& message) {
    return queue->tryPush(message);
  }
  bool tryPop(TtsAudioChunkMessage& out) { return queue->tryPop(out); }
  bool blockingPop(TtsAudioChunkMessage& out) {
    queue->receive(out);
    return !out.isDone();
  }
  void shutdown() { queue->push(TtsAudioChunkMessage::done()); }
  void remove() { queue->remove(); }

 private:
  std::unique_ptr<Queue> queue;
};

class TtsQueueSet {
 public:
  std::shared_ptr<TtsTaskQueue> taskQueue;
  std::vector<std::shared_ptr<TtsAudioChunkQueue>> audioQueues;
  std::vector<std::shared_ptr<tt::ipc::ICancelQueue>> cancelQueues;
  TtsQueueNames names;

  TtsQueueSet(int numWorkers, const config::TtsConfig& config,
              TtsQueueNames queueNames = {})
      : names(std::move(queueNames)) {
    taskQueue = std::make_shared<TtsTaskQueue>(
        names.taskQueue, static_cast<int>(config.taskQueueCapacity));
    audioQueues.reserve(numWorkers);
    cancelQueues.reserve(numWorkers);
    for (int i = 0; i < numWorkers; ++i) {
      audioQueues.emplace_back(std::make_shared<TtsAudioChunkQueue>(
          names.audioQueuePrefix + std::to_string(i),
          static_cast<int>(config.audioQueueCapacity)));
      cancelQueues.emplace_back(std::make_shared<tt::ipc::boost::CancelQueue>(
          names.cancelQueuePrefix + std::to_string(i),
          static_cast<int>(config.cancelQueueCapacity)));
    }
  }

  ~TtsQueueSet() { clear(); }

  void clear() {
    if (taskQueue) {
      taskQueue->remove();
      taskQueue.reset();
    }
    for (auto& queue : audioQueues) {
      queue->shutdown();
      queue->remove();
    }
    audioQueues.clear();
    for (auto& queue : cancelQueues) {
      queue->remove();
    }
    cancelQueues.clear();
  }

  TtsQueueSet(const TtsQueueSet&) = delete;
  TtsQueueSet& operator=(const TtsQueueSet&) = delete;

  TtsQueueSet(TtsQueueSet&&) = default;
  TtsQueueSet& operator=(TtsQueueSet&&) = default;
};

}  // namespace tt::ipc::tts
