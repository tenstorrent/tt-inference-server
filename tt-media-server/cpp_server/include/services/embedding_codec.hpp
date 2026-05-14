// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "domain/embedding_request.hpp"
#include "domain/embedding_response.hpp"

namespace tt::services::embedding_codec {

// Wire format (little-endian, native byte order):
//   [num_responses : uint32_t]
//   Per response:
//     [task_id      : uint32_t]
//     [has_error    : uint8_t]
//     If has_error:
//       [error_len  : uint32_t][error : chars]
//     Else:
//       [dim        : uint32_t][embedding : dim × float]
//       [total_tokens : int32_t]
//       [model_len  : uint32_t][model : chars]

namespace detail {

inline void appendRaw(std::vector<uint8_t>& buf, const void* data,
                      size_t bytes) {
  const auto* p = static_cast<const uint8_t*>(data);
  buf.insert(buf.end(), p, p + bytes);
}

template <typename T>
void appendScalar(std::vector<uint8_t>& buf, T val) {
  appendRaw(buf, &val, sizeof(val));
}

inline void appendString(std::vector<uint8_t>& buf, const std::string& s) {
  appendScalar(buf, static_cast<uint32_t>(s.size()));
  buf.insert(buf.end(), s.begin(), s.end());
}

inline void appendFloats(std::vector<uint8_t>& buf,
                         const std::vector<float>& v) {
  appendScalar(buf, static_cast<uint32_t>(v.size()));
  appendRaw(buf, v.data(), v.size() * sizeof(float));
}

class Reader {
 public:
  Reader(const uint8_t* data, size_t size) : data_(data), size_(size) {}

  uint32_t readUint32() {
    uint32_t v;
    std::memcpy(&v, data_ + off_, sizeof(v));
    off_ += sizeof(v);
    return v;
  }

  int32_t readInt32() {
    int32_t v;
    std::memcpy(&v, data_ + off_, sizeof(v));
    off_ += sizeof(v);
    return v;
  }

  uint8_t readUint8() { return data_[off_++]; }

  std::string readString() {
    uint32_t len = readUint32();
    std::string s(reinterpret_cast<const char*>(data_ + off_), len);
    off_ += len;
    return s;
  }

  std::vector<float> readFloats() {
    uint32_t count = readUint32();
    std::vector<float> v(count);
    std::memcpy(v.data(), data_ + off_, count * sizeof(float));
    off_ += count * sizeof(float);
    return v;
  }

  bool atEnd() const { return off_ >= size_; }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t off_ = 0;
};

}  // namespace detail

inline std::vector<uint8_t> encodeResponses(
    const std::vector<domain::EmbeddingRequest>& batch,
    const std::vector<domain::EmbeddingResponse>& responses) {
  std::vector<uint8_t> buf;
  buf.reserve(batch.size() * 4200);

  detail::appendScalar(buf, static_cast<uint32_t>(batch.size()));

  for (size_t i = 0; i < batch.size(); ++i) {
    detail::appendScalar(buf, batch[i].task_id);

    if (i < responses.size() && responses[i].error.empty()) {
      buf.push_back(0);
      detail::appendFloats(buf, responses[i].embedding);
      detail::appendScalar(buf,
                           static_cast<int32_t>(responses[i].total_tokens));
      detail::appendString(buf, responses[i].model);
    } else {
      buf.push_back(1);
      std::string error = (i < responses.size()) ? responses[i].error
                                                 : "No response from runner";
      detail::appendString(buf, error);
    }
  }
  return buf;
}

inline std::unordered_map<uint32_t, domain::EmbeddingResponse> decodeResponses(
    const std::vector<uint8_t>& buffer) {
  detail::Reader r(buffer.data(), buffer.size());
  uint32_t count = r.readUint32();

  std::unordered_map<uint32_t, domain::EmbeddingResponse> map;
  map.reserve(count);

  for (uint32_t i = 0; i < count && !r.atEnd(); ++i) {
    domain::EmbeddingResponse resp{r.readUint32()};
    uint8_t hasError = r.readUint8();

    if (hasError) {
      resp.error = r.readString();
    } else {
      resp.embedding = r.readFloats();
      resp.total_tokens = r.readInt32();
      resp.model = r.readString();
    }
    map.insert_or_assign(resp.task_id, std::move(resp));
  }
  return map;
}

}  // namespace tt::services::embedding_codec
