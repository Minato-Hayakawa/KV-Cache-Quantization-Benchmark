#ifndef KVQ_COMMON_HPP
#define KVQ_COMMON_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

namespace kvq {

constexpr int HEAD_DIM = 128;

struct QuantResult {
    std::string method_name;
    double encode_time_us;
    double decode_time_us;
    double cosine_sim;
    double logit_mae;
};

inline float dot_product(const float* a, const float* b, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) sum += a[i] * b[i];
    return sum;
}

inline float norm(const float* vec, int size) {
    return std::sqrt(dot_product(vec, vec, size));
}

inline float cosine_similarity(const float* a, const float* b, int size) {
    float n_a = norm(a, size);
    float n_b = norm(b, size);
    if (n_a == 0.0f || n_b == 0.0f) return 0.0f;
    return dot_product(a, b, size) / (n_a * n_b);
}

} // namespace kvq

#endif // KVQ_COMMON_HPP