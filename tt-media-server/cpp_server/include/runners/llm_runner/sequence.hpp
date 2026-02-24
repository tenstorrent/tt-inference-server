#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include "runners/llm_runner/sampling_params.hpp"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

namespace llm_engine {

struct TaskID {
  static constexpr size_t kSerializedSize = 36;
  TaskID() {
    auto uuid = boost::uuids::random_generator()();
    id = boost::uuids::to_string(uuid);
  }
  std::string id;

  bool operator==(const TaskID& other) const {
    return id == other.id;
  }

  std::vector<char> serialize() const {
    std::vector<char> buf(kSerializedSize, '\0');
    std::copy_n(id.begin(), std::min(id.size(), kSerializedSize), buf.begin());
    return buf;
  }

  static TaskID deserialize(const char* data, size_t len) {
    TaskID tid;
    size_t actual_len = strnlen(data, len);
    tid.id = std::string(data, actual_len);
    return tid;
  }
};

inline std::ostream& operator<<(std::ostream& os, const TaskID& tid) {
  return os << tid.id;
}

enum class SequenceStatus { WAITING, RUNNING, IN_FLIGHT, FINISHED };

struct TokenResult {
  TaskID task_id;
  uint64_t token_id;
  std::optional<bool> finished;
  bool is_stop_token = false;
  bool is_error = false;
};

class Sequence {
 public:
  static constexpr int block_size = 32;

  Sequence(std::vector<int64_t> token_ids,
           const SamplingParams& sampling_params = SamplingParams());

  void serialize(std::ostream& os) const;

  static Sequence* deserialize(std::istream& is);

  size_t size() const { return token_ids_.size(); }
  int64_t operator[](size_t i) const { return token_ids_[i]; }

  bool is_finished() const { return status_ == SequenceStatus::FINISHED; }
  size_t num_completion_tokens() const {
    return token_ids_.size() - num_prompt_tokens_;
  }
  size_t num_cached_blocks() const { return num_cached_tokens_ / block_size; }
  size_t num_blocks() const {
    return (token_ids_.size() + block_size - 1) / block_size;
  }
  int last_block_num_tokens() const {
    return static_cast<int>(token_ids_.size()) -
           static_cast<int>(num_blocks() - 1) * block_size;
  }

  std::vector<int64_t> block(size_t i) const;
  std::vector<int64_t> completion_token_ids() const;

  void append_token(int64_t token_id);

  TaskID task_id;
  SequenceStatus status_ = SequenceStatus::WAITING;
  std::vector<int64_t> token_ids_;
  int64_t last_token = 0;
  size_t num_prompt_tokens_ = 0;
  size_t num_cached_tokens_ = 0;
  std::vector<int> block_table_;
  std::unique_ptr<SamplingParams> sampling_params;

 private:
  size_t num_tokens() const { return token_ids_.size(); }
};

}  // namespace llm_engine

namespace std {
  template <>
  struct hash<llm_engine::TaskID> {
    size_t operator()(const llm_engine::TaskID& s) const {
      return hash<string>{}(s.id);
    }
  };
  }  // namespace std
