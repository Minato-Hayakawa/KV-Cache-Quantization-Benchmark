#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include "common.hpp"

namespace kvq {

class RotorQuant {
public:
    RotorQuant(size_t dim, int num_bits = 3, size_t block_size = 4)
        : dim_(dim), num_bits_(num_bits), block_size_(block_size) {
        assert(dim_ > 0);
        assert(num_bits_ >= 2 && num_bits_ <= 8);
        assert(block_size_ >= 2);
    }

    // Clifford/Givensロータ（ブロック単位の回転）を適用して量子化
    std::vector<int8_t> compress(const std::vector<float>& input, float& scale) const {
        assert(input.size() == dim_);
        std::vector<float> rotated(dim_);

        constexpr float cos_t = 0.7071067811865475f; // 1 / sqrt(2)
        constexpr float sin_t = 0.7071067811865475f;

        // ブロックごとにローカル2D Givens/Rotor回転を適用
        for (size_t b = 0; b < dim_; b += block_size_) {
            size_t block_end = std::min(b + block_size_, dim_);
            size_t i = b;

            // 2要素ずつペアにして回転
            for (; i + 1 < block_end; i += 2) {
                size_t idx1 = i;
                size_t idx2 = i + 1;

                rotated[idx1] = cos_t * input[idx1] - sin_t * input[idx2];
                rotated[idx2] = sin_t * input[idx1] + cos_t * input[idx2];
            }

            // 余った1要素がある場合は回転させずにそのまま渡す（0埋め防止）
            if (i < block_end) {
                rotated[i] = input[i];
            }
        }

        // 量子化処理
        float max_val = 0.0f;
        for (float v : rotated) max_val = std::max(max_val, std::abs(v));

        int qmax = (1 << (num_bits_ - 1)) - 1;
        scale = max_val / static_cast<float>(qmax);
        if (scale == 0.0f) scale = 1.0f;

        std::vector<int8_t> quantized(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            int q = static_cast<int>(std::round(rotated[i] / scale));
            quantized[i] = static_cast<int8_t>(std::clamp(q, -qmax, qmax));
        }

        return quantized;
    }

    std::vector<float> decompress(const std::vector<int8_t>& quantized, float scale) const {
        assert(quantized.size() == dim_);
        std::vector<float> rotated_deq(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            rotated_deq[i] = static_cast<float>(quantized[i]) * scale;
        }

        std::vector<float> output(dim_);
        constexpr float cos_t = 0.7071067811865475f;
        constexpr float sin_t = 0.7071067811865475f;

        // 逆Rotor回転 (R^T)
        for (size_t b = 0; b < dim_; b += block_size_) {
            size_t block_end = std::min(b + block_size_, dim_);
            size_t i = b;

            for (; i + 1 < block_end; i += 2) {
                size_t idx1 = i;
                size_t idx2 = i + 1;

                output[idx1] =  cos_t * rotated_deq[idx1] + sin_t * rotated_deq[idx2];
                output[idx2] = -sin_t * rotated_deq[idx1] + cos_t * rotated_deq[idx2];
            }

            if (i < block_end) {
                output[i] = rotated_deq[i];
            }
        }

        return output;
    }

private:
    size_t dim_;
    int num_bits_;
    size_t block_size_;
};

} // namespace kvq