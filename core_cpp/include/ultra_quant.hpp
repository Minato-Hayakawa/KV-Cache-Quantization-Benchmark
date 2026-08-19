#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include "common.hpp"

namespace kvq {

class UltraQuant {
public:
    explicit UltraQuant(size_t dim) : dim_(dim) {
        assert(dim_ > 0);
        // FWHTの要件：次元数が2のべき乗であることの確認
        assert((dim_ & (dim_ - 1)) == 0 && "UltraQuant (FWHT) requires dimension to be a power of 2.");
    }

    // 高速Walsh-Hadamard変換 (FWHT: O(d log d))
    static void fwht(std::vector<float>& data) {
        size_t n = data.size();
        // 念のため実行時にもサイズチェック
        assert((n & (n - 1)) == 0 && "FWHT size must be a power of 2.");

        for (size_t len = 1; len < n; len <<= 1) {
            for (size_t i = 0; i < n; i += 2 * len) {
                for (size_t j = 0; j < len; ++j) {
                    float u = data[i + j];
                    float v = data[i + len + j];
                    data[i + j] = u + v;
                    data[i + len + j] = u - v;
                }
            }
        }
        float norm = 1.0f / std::sqrt(static_cast<float>(n));
        for (size_t i = 0; i < n; ++i) {
            data[i] *= norm;
        }
    }

    // FP4 (E2M1) グリッド値への直接マッピング (16ステップ)
    std::vector<uint8_t> compress(const std::vector<float>& input, float& block_scale) const {
        assert(input.size() == dim_ && "Input dimension mismatch.");
        std::vector<float> transformed = input;
        fwht(transformed); // WHT回転による外れ値分散

        // 全体（またはブロック単位）のスケール算出
        float max_val = 0.0f;
        for (float v : transformed) {
            max_val = std::max(max_val, std::abs(v));
        }
        
        constexpr float fp4_max = 6.0f; // E2M1 最大表現値
        block_scale = max_val / fp4_max;
        if (block_scale == 0.0f) block_scale = 1.0f;

        float inv_block_scale = 1.0f / block_scale;
        std::vector<uint8_t> fp4_indices(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            float val = transformed[i] * inv_block_scale;
            fp4_indices[i] = quantize_fp4_e2m1(val);
        }

        return fp4_indices;
    }

    std::vector<float> decompress(const std::vector<uint8_t>& fp4_indices, float block_scale) const {
        assert(fp4_indices.size() == dim_ && "FP4 indices dimension mismatch.");
        std::vector<float> transformed(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            transformed[i] = dequantize_fp4_e2m1(fp4_indices[i]) * block_scale;
        }

        fwht(transformed); // FWHTは自己逆変換 (WHT = WHT^-1)
        return transformed;
    }

private:
    size_t dim_;

    // FP4 (E2M1) グリッド定数テーブル
    static constexpr std::array<float, 16> kFP4Table = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    uint8_t quantize_fp4_e2m1(float val) const {
        uint8_t best_idx = 0;
        float min_diff = 1e9f;
        for (uint8_t i = 0; i < 16; ++i) {
            float diff = std::abs(val - kFP4Table[i]);
            if (diff < min_diff) {
                min_diff = diff;
                best_idx = i;
            }
        }
        return best_idx;
    }

    float dequantize_fp4_e2m1(uint8_t idx) const {
        return kFP4Table[idx & 0x0F];
    }
};

} // namespace kvq