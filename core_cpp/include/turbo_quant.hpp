#pragma once

#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include "common.hpp"

namespace kvq {

class TurboQuant {
public:
    TurboQuant(size_t dim, int num_bits = 3, uint32_t seed = 42)
        : dim_(dim), num_bits_(num_bits), rotation_matrix_(dim * dim, 0.0f) {
        assert(dim_ > 0);
        assert(num_bits_ >= 2 && num_bits_ <= 8);
        generate_orthogonal_matrix(seed);
    }

    // Keyベクトルの回転と均一スカラー量子化
    std::vector<int8_t> compress(const std::vector<float>& input, float& scale) const {
        assert(input.size() == dim_ && "Input vector dimension mismatch.");
        std::vector<float> rotated(dim_, 0.0f);
        
        // 1. 全体直交回転: rotated = R * input
        for (size_t i = 0; i < dim_; ++i) {
            const float* row = &rotation_matrix_[i * dim_];
            float sum = 0.0f;
            for (size_t j = 0; j < dim_; ++j) {
                sum += row[j] * input[j];
            }
            rotated[i] = sum;
        }

        // 2. スケール計算 & 量子化
        float max_val = 0.0f;
        for (float v : rotated) max_val = std::max(max_val, std::abs(v));
        
        int qmax = (1 << (num_bits_ - 1)) - 1;
        scale = max_val / static_cast<float>(qmax);
        if (scale == 0.0f) scale = 1.0f;

        float inv_scale = 1.0f / scale;
        std::vector<int8_t> quantized(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            int q = static_cast<int>(std::round(rotated[i] * inv_scale));
            quantized[i] = static_cast<int8_t>(std::clamp(q, -qmax, qmax));
        }

        return quantized;
    }

    // 逆量子化と逆回転
    std::vector<float> decompress(const std::vector<int8_t>& quantized, float scale) const {
        assert(quantized.size() == dim_ && "Quantized vector dimension mismatch.");
        std::vector<float> rotated_deq(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            rotated_deq[i] = static_cast<float>(quantized[i]) * scale;
        }

        std::vector<float> output(dim_, 0.0f);

        // 3. 逆回転: output = R^T * rotated_deq
        // メモリの連続アクセスのためループ順序を最適化 (j -> i)
        for (size_t j = 0; j < dim_; ++j) {
            const float* row = &rotation_matrix_[j * dim_];
            float val = rotated_deq[j];
            for (size_t i = 0; i < dim_; ++i) {
                output[i] += row[i] * val;
            }
        }

        return output;
    }

private:
    size_t dim_;
    int num_bits_;
    std::vector<float> rotation_matrix_;

    // 複数のHouseholder変換を合成して密な直交行列（Haar分布に近い）を生成
    void generate_orthogonal_matrix(uint32_t seed) {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        // 単位行列で初期化
        for (size_t i = 0; i < dim_; ++i) {
            rotation_matrix_[i * dim_ + i] = 1.0f;
        }

        // k個のランダムHouseholder鏡映を連続適用して回転を合成
        size_t num_reflections = std::min(dim_, static_size_t(8)); // 次元に応じた鏡映数
        for (size_t r = 0; r < num_reflections; ++r) {
            std::vector<float> v(dim_);
            float norm_sq = 0.0f;
            for (size_t i = 0; i < dim_; ++i) {
                v[i] = dist(gen);
                norm_sq += v[i] * v[i];
            }
            if (norm_sq < 1e-8f) continue;

            // H = I - 2 * v * v^T / norm_sq
            // rotation_matrix = H * rotation_matrix
            std::vector<float> temp(dim_ * dim_, 0.0f);
            for (size_t i = 0; i < dim_; ++i) {
                // v^T * R_col
                std::vector<float> v_dot_R(dim_, 0.0f);
                for (size_t j = 0; j < dim_; ++j) {
                    for (size_t k = 0; k < dim_; ++k) {
                        v_dot_R[j] += v[k] * rotation_matrix_[k * dim_ + j];
                    }
                }
                for (size_t j = 0; j < dim_; ++j) {
                    temp[i * dim_ + j] = rotation_matrix_[i * dim_ + j] - (2.0f * v[i] / norm_sq) * v_dot_R[j];
                }
            }
            rotation_matrix_ = std::move(temp);
        }
    }

    static constexpr size_t static_size_t(size_t val) { return val; }
};

} // namespace kvq