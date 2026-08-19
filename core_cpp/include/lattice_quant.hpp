#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <limits>
#include "common.hpp"
#include "ultra_quant.hpp"

namespace kvq {

class LatticeQuant {
public:
    explicit LatticeQuant(size_t dim, int num_bits = 3)
        : dim_(dim), num_bits_(num_bits) {
        assert(dim % 4 == 0 && "dim must be a multiple of 4 for D4 lattice quantization.");
        assert(num_bits > 1 && "num_bits must be greater than 1.");
    }

    std::vector<int16_t> compress(const std::vector<float>& input, float& scale) const {
        assert(input.size() == dim_ && "Input vector dimension mismatch.");

        std::vector<float> transformed = input;
        UltraQuant::fwht(transformed); // RHT (Random Hadamard Transform)

        float max_val = 0.0f;
        for (float v : transformed) {
            max_val = std::max(max_val, std::abs(v));
        }

        int qmax = (1 << (num_bits_ - 1)) - 1;
        scale = max_val / static_cast<float>(qmax);
        if (scale == 0.0f) scale = 1.0f;

        std::vector<int16_t> lattice_points(dim_);

        for (size_t i = 0; i < dim_; i += 4) {
            float f[4] = {
                transformed[i] / scale,
                transformed[i+1] / scale,
                transformed[i+2] / scale,
                transformed[i+3] / scale
            };

            int r[4];
            // 1. 最初にあらかじめ境界内にクランプして丸める
            for (int k = 0; k < 4; ++k) {
                r[k] = std::clamp(static_cast<int>(std::round(f[k])), -qmax, qmax);
            }

            // 2. D4 Lattice条件判定: x1 + x2 + x3 + x4 ≡ 0 (mod 2)
            if (std::abs(r[0] + r[1] + r[2] + r[3]) % 2 != 0) {
                int best_k = -1;
                int best_step = 0;
                float min_cost_increase = std::numeric_limits<float>::infinity();

                // 範囲内(-qmax ~ qmax)に収まるステップの中で、二乗誤差増加が最も少ないものを探索
                for (int k = 0; k < 4; ++k) {
                    for (int step : {1, -1}) {
                        int next_r = r[k] + step;
                        if (next_r >= -qmax && next_r <= qmax) {
                            float orig_err = f[k] - static_cast<float>(r[k]);
                            float new_err  = f[k] - static_cast<float>(next_r);
                            float cost_increase = (new_err * new_err) - (orig_err * orig_err);

                            if (cost_increase < min_cost_increase) {
                                min_cost_increase = cost_increase;
                                best_k = k;
                                best_step = step;
                            }
                        }
                    }
                }

                if (best_k != -1) {
                    r[best_k] += best_step;
                }
            }

            lattice_points[i]   = static_cast<int16_t>(r[0]);
            lattice_points[i+1] = static_cast<int16_t>(r[1]);
            lattice_points[i+2] = static_cast<int16_t>(r[2]);
            lattice_points[i+3] = static_cast<int16_t>(r[3]);
        }

        return lattice_points;
    }

    std::vector<float> decompress(const std::vector<int16_t>& lattice_points, float scale) const {
        assert(lattice_points.size() == dim_ && "Lattice points vector dimension mismatch.");

        std::vector<float> transformed(dim_);
        for (size_t i = 0; i < dim_; ++i) {
            transformed[i] = static_cast<float>(lattice_points[i]) * scale;
        }

        UltraQuant::fwht(transformed); // 逆RHT
        return transformed;
    }

private:
    size_t dim_;
    int num_bits_;
};

} // namespace kvq