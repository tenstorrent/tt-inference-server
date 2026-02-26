#include "runners/llm_runner/sampling_params.hpp"

namespace llm_engine {

namespace {

template <typename T>
void write_scalar(std::ostream& os, const T& value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_scalar(std::istream& is) {
    T value;
    is.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

template <typename T>
void write_optional(std::ostream& os, const std::optional<T>& opt) {
    bool has_value = opt.has_value();
    write_scalar(os, has_value);
    if (has_value) {
        write_scalar(os, *opt);
    }
}

template <typename T>
std::optional<T> read_optional(std::istream& is) {
    if (read_scalar<bool>(is)) {
        return read_scalar<T>(is);
    }
    return std::nullopt;
}

template <typename T>
void write_vector(std::ostream& os, const std::vector<T>& vec) {
    size_t size = vec.size();
    write_scalar(os, size);
    os.write(reinterpret_cast<const char*>(vec.data()), size * sizeof(T));
}

template <typename T>
std::vector<T> read_vector(std::istream& is) {
    size_t size = read_scalar<size_t>(is);
    std::vector<T> vec(size);
    is.read(reinterpret_cast<char*>(vec.data()), size * sizeof(T));
    return vec;
}

} // anonymous namespace

void SamplingParams::serialize(std::ostream& os) const {
    write_scalar(os, temperature);
    write_scalar(os, max_tokens);
    write_scalar(os, ignore_eos);
    write_optional(os, top_p);
    write_scalar(os, presence_penalty);
    write_scalar(os, frequency_penalty);
    write_optional(os, seed);
    write_scalar(os, use_beam_search);
    write_optional(os, top_k);
    write_optional(os, min_p);
    write_optional(os, repetition_penalty);
    write_scalar(os, length_penalty);
    write_vector(os, stop_token_ids);
    write_scalar(os, include_stop_str_in_output);
    write_scalar(os, min_tokens);
    write_scalar(os, skip_special_tokens);
    write_scalar(os, spaces_between_special_tokens);

    bool has_allowed = allowed_token_ids.has_value();
    write_scalar(os, has_allowed);
    if (has_allowed) {
        write_vector(os, *allowed_token_ids);
    }

    write_optional(os, prompt_logprobs);
    write_optional(os, truncate_prompt_tokens);
}

SamplingParams* SamplingParams::deserialize(std::istream& is) {
    auto* params = new SamplingParams();

    params->temperature               = read_scalar<float>(is);
    params->max_tokens                = read_scalar<int>(is);
    params->ignore_eos                = read_scalar<bool>(is);
    params->top_p                     = read_optional<float>(is);
    params->presence_penalty          = read_scalar<float>(is);
    params->frequency_penalty         = read_scalar<float>(is);
    params->seed                      = read_optional<int>(is);
    params->use_beam_search           = read_scalar<bool>(is);
    params->top_k                     = read_optional<int>(is);
    params->min_p                     = read_optional<float>(is);
    params->repetition_penalty        = read_optional<float>(is);
    params->length_penalty            = read_scalar<float>(is);
    params->stop_token_ids            = read_vector<int>(is);
    params->include_stop_str_in_output = read_scalar<bool>(is);
    params->min_tokens                = read_scalar<int>(is);
    params->skip_special_tokens       = read_scalar<bool>(is);
    params->spaces_between_special_tokens = read_scalar<bool>(is);

    if (read_scalar<bool>(is)) {
        params->allowed_token_ids = read_vector<int>(is);
    }

    params->prompt_logprobs           = read_optional<int>(is);
    params->truncate_prompt_tokens    = read_optional<int>(is);

    return params;
}

} // namespace llm_engine