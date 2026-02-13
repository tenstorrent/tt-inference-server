#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
#include "api/alignment.h"
#include "llm_engine/sampling_params.hpp"

#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>





namespace llm_engine {
  
struct SequenceID {
  static constexpr size_t kSerializedSize = 36;

  SequenceID() {
    auto uuid = boost::uuids::random_generator()();
    id = boost::uuids::to_string(uuid);
  }
  std::string id;

  bool operator==(const SequenceID& other) const {
    return id == other.id;
  }

  std::vector<char> serialize() const {
    std::vector<char> buf(kSerializedSize, '\0');
    std::copy_n(id.begin(), std::min(id.size(), kSerializedSize), buf.begin());
    return buf;
  }

  static SequenceID deserialize(const char* data, size_t len) {
    SequenceID sid;
    sid.id = std::string(data, len);
    return sid;
  }
};

inline std::ostream& operator<<(std::ostream& os, const SequenceID& sid) {
  return os << sid.id;
}


  

enum class SequenceStatus { WAITING, RUNNING, IN_FLIGHT, FINISHED };

class Sequence {
 public:
  static constexpr int block_size = 256;
  static int next_seq_id();

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

  static constexpr size_t h2d_payload_size() {
    return SequenceID::kSerializedSize + sizeof(int64_t);
  }

  static uint32_t page_size() {
    return align(h2d_payload_size(), 64);
  }

  std::vector<char> to_h2d_input() const {
    std::vector<char> input(page_size(), 0);
    auto id_bytes = seq_id.serialize();
    std::copy(id_bytes.begin(), id_bytes.end(), input.begin());
    std::memcpy(input.data() + SequenceID::kSerializedSize, &last_token, sizeof(last_token));
    return input;
  }

  static Sequence* from_h2d_input(const std::vector<char>& input) {
    Sequence* seq = new Sequence(std::vector<int64_t>{});
    seq->seq_id = SequenceID::deserialize(input.data(), SequenceID::kSerializedSize);
    seq->last_token = *reinterpret_cast<const int64_t*>(input.data() + SequenceID::kSerializedSize);
    return seq;
  }

  std::vector<int64_t> block(size_t i) const;
  std::vector<int64_t> completion_token_ids() const;

  void append_token(int64_t token_id);

  SequenceID seq_id;
  SequenceStatus status_ = SequenceStatus::WAITING;
  std::vector<int64_t> token_ids_;
  int64_t last_token = 0;
  size_t num_prompt_tokens_ = 0;
  size_t num_cached_tokens_ = 0;
  std::vector<int> block_table_;
  float temperature = 1.0f;
  /** Max completion tokens for this sequence (from SamplingParams). Each request can have a different value. */
  int max_tokens = 64;
  bool ignore_eos = false;

 private:
  size_t num_tokens() const { return token_ids_.size(); }
};

}  // namespace llm_engine

namespace std {
  template <>
  struct hash<llm_engine::SequenceID> {
    size_t operator()(const llm_engine::SequenceID& s) const {
      return hash<string>{}(s.id);
    }
  };
  }  // namespace std