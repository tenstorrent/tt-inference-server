#include "runners/llm_runner/sampling_params.hpp"

namespace llm_engine {

namespace {

template <typename T>
void writeScalar(std::ostream& os, const T& value) {
  os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T readScalar(std::istream& is) {
  T value;
  is.read(reinterpret_cast<char*>(&value), sizeof(T));
  return value;
}

template <typename T>
void writeOptional(std::ostream& os, const std::optional<T>& opt) {
  bool hasValue = opt.has_value();
  writeScalar(os, hasValue);
  if (hasValue) {
    writeScalar(os, *opt);
  }
}

template <typename T>
std::optional<T> readOptional(std::istream& is) {
  if (readScalar<bool>(is)) {
    return readScalar<T>(is);
  }
  return std::nullopt;
}

template <typename T>
void writeVector(std::ostream& os, const std::vector<T>& vec) {
  size_t size = vec.size();
  writeScalar(os, size);
  os.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
}

template <typename T>
std::vector<T> readVector(std::istream& is) {
  size_t size = readScalar<size_t>(is);
  std::vector<T> vec(size);
  is.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
  return vec;
}

}  // anonymous namespace

void SamplingParams::serialize(std::ostream& os) const {
  writeScalar(os, temperature);
  writeOptional(os, max_tokens);
  writeScalar(os, ignore_eos);
  writeOptional(os, top_p);
  writeScalar(os, presence_penalty);
  writeScalar(os, frequency_penalty);
  writeOptional(os, seed);
  writeScalar(os, use_beam_search);
  writeOptional(os, top_k);
  writeOptional(os, min_p);
  writeOptional(os, repetition_penalty);
  writeScalar(os, length_penalty);
  writeVector(os, stop_token_ids);
  writeScalar(os, include_stop_str_in_output);
  writeScalar(os, min_tokens);
  writeScalar(os, skip_special_tokens);
  writeScalar(os, spaces_between_special_tokens);
  writeScalar(os, fast_mode);

  bool hasAllowed = allowed_token_ids.has_value();
  writeScalar(os, hasAllowed);
  if (hasAllowed) {
    writeVector(os, *allowed_token_ids);
  }

  writeOptional(os, prompt_logprobs);
  writeOptional(os, truncate_prompt_tokens);
}

SamplingParams* SamplingParams::deserialize(std::istream& is) {
  auto* params = new SamplingParams();

  params->temperature = readScalar<float>(is);
  params->max_tokens = readOptional<int>(is);
  params->ignore_eos = readScalar<bool>(is);
  params->top_p = readOptional<float>(is);
  params->presence_penalty = readScalar<float>(is);
  params->frequency_penalty = readScalar<float>(is);
  params->seed = readOptional<int>(is);
  params->use_beam_search = readScalar<bool>(is);
  params->top_k = readOptional<int>(is);
  params->min_p = readOptional<float>(is);
  params->repetition_penalty = readOptional<float>(is);
  params->length_penalty = readScalar<float>(is);
  params->stop_token_ids = readVector<int>(is);
  params->include_stop_str_in_output = readScalar<bool>(is);
  params->min_tokens = readScalar<int>(is);
  params->skip_special_tokens = readScalar<bool>(is);
  params->spaces_between_special_tokens = readScalar<bool>(is);
  params->fast_mode = readScalar<bool>(is);

  if (readScalar<bool>(is)) {
    params->allowed_token_ids = readVector<int>(is);
  }

  params->prompt_logprobs = readOptional<int>(is);
  params->truncate_prompt_tokens = readOptional<int>(is);

  return params;
}

}  // namespace llm_engine