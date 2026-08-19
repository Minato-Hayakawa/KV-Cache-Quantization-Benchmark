#ifndef KVQ_ROPE_TQ_HPP
#define KVQ_ROPE_TQ_HPP

#include <vector>
#include <cstdint>
#include "common.hpp"

namespace kvq {

class RoPEAwareTurboQuant {
public:
    explicit RoPEAwareTurboQuant(int dim = HEAD_DIM, float rope_base = 10000.0f);

    std::vector<int8_t> compress(const std::vector<float>& input, int pos, float& scale) const;
    std::vector<float> decompress(const std::vector<int8_t>& quantized, int pos, float scale) const;

private:
    int dim_;
    float rope_base_;
    std::vector<float> inv_freq_; // 事前計算用 1.0 / (rope_base ^ (2i / dim))

    void apply_rope(const float* in, float* out, int pos, bool inverse) const;
};

} // namespace kvq

#endif // KVQ_ROPE_TQ_HPP