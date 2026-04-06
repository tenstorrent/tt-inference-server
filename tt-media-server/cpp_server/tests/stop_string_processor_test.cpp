// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

#include "services/stop_string_processor.hpp"

#include <gtest/gtest.h>

#include <thread>
#include <vector>

using namespace tt::services;

class StopStringProcessorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    cancel_called_ = false;
    cancelled_task_ = 0;
    cancel_count_ = 0;
  }

  StopStringProcessor::CancelCallback makeCallback() {
    return [this](uint32_t taskId) {
      cancel_called_ = true;
      cancelled_task_ = taskId;
      cancel_count_++;
    };
  }

  bool cancel_called_;
  uint32_t cancelled_task_;
  int cancel_count_;
};

TEST_F(StopStringProcessorTest, ConstructorRequiresCallback) {
  EXPECT_THROW(StopStringProcessor(nullptr), std::invalid_argument);
}

TEST_F(StopStringProcessorTest, SimpleStopStringMatch) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"STOP"});
  EXPECT_EQ(processor.activeTaskCount(), 1);

  auto result1 = processor.processText(1, "Hello ");
  EXPECT_FALSE(result1.stop_detected);
  EXPECT_EQ(result1.output_text, "Hello ");
  EXPECT_FALSE(cancel_called_);

  auto result2 = processor.processText(1, "STOP");
  EXPECT_TRUE(result2.stop_detected);
  EXPECT_EQ(result2.matched_string, "STOP");
  EXPECT_EQ(result2.output_text,
            "Hello ");  // Accumulated text with stop removed
  EXPECT_TRUE(cancel_called_);
  EXPECT_EQ(cancelled_task_, 1);
}

TEST_F(StopStringProcessorTest, MultiCharacterSequence) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"<|endoftext|>"});

  auto result1 = processor.processText(1, "Some text ");
  EXPECT_FALSE(result1.stop_detected);

  auto result2 = processor.processText(1, "<|endoftext|>");
  EXPECT_TRUE(result2.stop_detected);
  EXPECT_EQ(result2.matched_string, "<|endoftext|>");
  EXPECT_TRUE(cancel_called_);
}

TEST_F(StopStringProcessorTest, MultipleStopSequences) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"\n\n", "END", "STOP"});

  auto result1 = processor.processText(1, "Text ");
  EXPECT_FALSE(result1.stop_detected);

  auto result2 = processor.processText(1, "more\n\n");
  EXPECT_TRUE(result2.stop_detected);
  EXPECT_EQ(result2.matched_string, "\n\n");
  EXPECT_TRUE(cancel_called_);
}

TEST_F(StopStringProcessorTest, PartialMatchAcrossCalls) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"STOP"});

  auto result1 = processor.processText(1, "Hello ST");
  EXPECT_FALSE(result1.stop_detected);

  auto result2 = processor.processText(1, "OP");
  EXPECT_TRUE(result2.stop_detected);
  EXPECT_EQ(result2.matched_string, "STOP");
  EXPECT_TRUE(cancel_called_);
}

TEST_F(StopStringProcessorTest, EmptyStopSequencesList) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {});

  auto result = processor.processText(1, "Any text STOP");
  EXPECT_FALSE(result.stop_detected);
  EXPECT_FALSE(cancel_called_);
}

TEST_F(StopStringProcessorTest, StopDetectedPersistence) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"STOP"});

  processor.processText(1, "Hello STOP");
  EXPECT_TRUE(processor.isStopDetected(1));

  // Subsequent calls should return empty
  auto result = processor.processText(1, " more text");
  EXPECT_TRUE(result.stop_detected);
  EXPECT_EQ(result.output_text, "");

  // Stop flag should persist
  EXPECT_TRUE(processor.isStopDetected(1));
}

TEST_F(StopStringProcessorTest, MultipleTaskIsolation) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"STOP1"});
  processor.initializeTask(2, {"STOP2"});
  EXPECT_EQ(processor.activeTaskCount(), 2);

  auto result1 = processor.processText(1, "Text STOP1");
  EXPECT_TRUE(result1.stop_detected);
  EXPECT_TRUE(processor.isStopDetected(1));

  // Task 2 should be unaffected
  EXPECT_FALSE(processor.isStopDetected(2));

  auto result2 = processor.processText(2, "Text STOP2");
  EXPECT_TRUE(result2.stop_detected);
  EXPECT_TRUE(processor.isStopDetected(2));

  EXPECT_EQ(cancel_count_, 2);
}

TEST_F(StopStringProcessorTest, Finalization) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"STOP"});
  EXPECT_EQ(processor.activeTaskCount(), 1);

  processor.finalizeTask(1);
  EXPECT_EQ(processor.activeTaskCount(), 0);

  // Should be safe to finalize again
  processor.finalizeTask(1);
  EXPECT_EQ(processor.activeTaskCount(), 0);
}

TEST_F(StopStringProcessorTest, UninitializedTaskHandling) {
  StopStringProcessor processor(makeCallback());

  // processText on uninitialized task should return no-match
  auto result = processor.processText(99, "text");
  EXPECT_FALSE(result.stop_detected);
  EXPECT_EQ(result.output_text, "text");

  // isStopDetected on uninitialized task should return false
  EXPECT_FALSE(processor.isStopDetected(99));
}

TEST_F(StopStringProcessorTest, NoMatchWithSimilarText) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"STOP"});

  auto result1 = processor.processText(1, "STOPPING");
  EXPECT_FALSE(result1.stop_detected);

  auto result2 = processor.processText(1, " STO");
  EXPECT_FALSE(result2.stop_detected);
}

TEST_F(StopStringProcessorTest, FirstMatchWins) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"STOP", "STOP HERE", "END"});

  auto result = processor.processText(1, "Text STOP");
  EXPECT_TRUE(result.stop_detected);
  // Should match "STOP" first even though "STOP HERE" could also match later
  EXPECT_EQ(result.matched_string, "STOP");
}

TEST_F(StopStringProcessorTest, ThreadSafety) {
  StopStringProcessor processor([](uint32_t) {});

  // Initialize multiple tasks
  for (uint32_t i = 0; i < 10; ++i) {
    processor.initializeTask(i, {"STOP"});
  }

  // Process text from multiple threads
  std::vector<std::thread> threads;
  for (uint32_t i = 0; i < 10; ++i) {
    threads.emplace_back([&processor, i]() {
      for (int j = 0; j < 100; ++j) {
        processor.processText(i, "text ");
      }
      processor.processText(i, "STOP");
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  // All tasks should have detected stop
  for (uint32_t i = 0; i < 10; ++i) {
    EXPECT_TRUE(processor.isStopDetected(i));
  }
}

TEST_F(StopStringProcessorTest, EmptyStopStringInList) {
  StopStringProcessor processor(makeCallback());

  // Empty strings in list should be skipped
  processor.initializeTask(1, {"", "STOP", ""});

  auto result1 = processor.processText(1, "text");
  EXPECT_FALSE(result1.stop_detected);

  auto result2 = processor.processText(1, "STOP");
  EXPECT_TRUE(result2.stop_detected);
  EXPECT_EQ(result2.matched_string, "STOP");
}

TEST_F(StopStringProcessorTest, ExactMatch) {
  StopStringProcessor processor(makeCallback());

  processor.initializeTask(1, {"STOP"});

  auto result = processor.processText(1, "STOP");
  EXPECT_TRUE(result.stop_detected);
  EXPECT_EQ(result.matched_string, "STOP");
  EXPECT_EQ(result.output_text, "");  // Everything was stop sequence
}
