// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cstdint>
#include <istream>
#include <ostream>
#include <span>
#include <streambuf>
#include <string>
#include <string_view>
#include <vector>

namespace tt::sockets::wire {

namespace detail {

/**
 * @brief std::streambuf that appends bytes straight into a
 * std::vector<uint8_t>.
 *
 * Lets Cereal serialize directly into the final output buffer, avoiding
 * std::ostringstream's reallocating growth plus the extra str()/vector copies
 * of the whole payload that the previous implementation paid on every send.
 */
class VectorOutStreamBuf : public std::streambuf {
 public:
  explicit VectorOutStreamBuf(std::vector<uint8_t>& out) : out_(out) {}

 protected:
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    const auto* bytes = reinterpret_cast<const uint8_t*>(s);
    out_.insert(out_.end(), bytes, bytes + n);
    return n;
  }

  int_type overflow(int_type ch) override {
    if (ch != traits_type::eof()) {
      out_.push_back(static_cast<uint8_t>(ch));
    }
    return ch;
  }

 private:
  std::vector<uint8_t>& out_;
};

/**
 * @brief Read-only std::streambuf over an existing byte span (zero-copy).
 *
 * Points Cereal's input archive straight at the received buffer so the payload
 * is never copied into an intermediate std::string + std::istringstream just to
 * be parsed. The span must outlive any archive reading from this buffer; the
 * archive only reads (sgetn/sbumpc) and never writes the get area, so the
 * const_cast below is safe.
 */
class SpanInStreamBuf : public std::streambuf {
 public:
  explicit SpanInStreamBuf(std::span<const uint8_t> data) {
    auto* base = const_cast<char*>(reinterpret_cast<const char*>(data.data()));
    setg(base, base, base + data.size());
  }
};

}  // namespace detail

template <typename T>
std::vector<uint8_t> serializeMessage(std::string_view messageType,
                                      const T& obj) {
  std::vector<uint8_t> out;
  detail::VectorOutStreamBuf buf(out);
  std::ostream os(&buf);
  {
    cereal::BinaryOutputArchive archive(os);
    std::string messageTypeString(messageType);
    archive(messageTypeString);
    obj.write(archive);
  }
  return out;
}

inline std::string readMessageType(std::span<const uint8_t> data) {
  detail::SpanInStreamBuf buf(data);
  std::istream is(&buf);

  cereal::BinaryInputArchive archive(is);
  std::string messageType;
  archive(messageType);
  return messageType;
}

template <typename T>
T deserializePayload(std::span<const uint8_t> data) {
  detail::SpanInStreamBuf buf(data);
  std::istream is(&buf);

  cereal::BinaryInputArchive archive(is);
  std::string ignoredMessageType;
  archive(ignoredMessageType);
  return T::read(archive);
}

}  // namespace tt::sockets::wire
