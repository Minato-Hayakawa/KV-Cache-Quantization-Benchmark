#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>

// include/ 配下の各手法のヘッダーをインクルード
#include "common.hpp"
#include "rope_tq.hpp"
#include "turbo_quant.hpp"
#include "rotor_quant.hpp"
#include "ultra_quant.hpp"
#include "lattice_quant.hpp"

// コサイン類似度の計算関数
float compute_cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

// 平均絶対誤差 (MAE) の計算関数
float compute_mae(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        err += std::abs(a[i] - b[i]);
    }
    return err / static_cast<float>(a.size());
}

int main() {
    constexpr size_t HEAD_DIM = 128;      // KV キャッシュの標準的なヘッド次元
    constexpr size_t NUM_VECTORS = 10000; // ベンチマーク用サンプル数
    constexpr int NUM_BITS = 3;           // 標準量子化ビット数 (3-bit)

    std::cout << "========================================================================\n";
    std::cout << "        KV-Cache Quantization C++ Microbenchmark Framework\n";
    std::cout << "        Vector Dim: " << HEAD_DIM << " | Samples: " << NUM_VECTORS << " | Bit-width: " << NUM_BITS << "-bit\n";
    std::cout << "========================================================================\n\n";

    // 1. ガウス分布に従う擬似 KV キャッシュデータの生成
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<std::vector<float>> dataset(NUM_VECTORS, std::vector<float>(HEAD_DIM));
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
        for (size_t d = 0; d < HEAD_DIM; ++d) {
            dataset[i][d] = dist(gen);
        }
    }

    // 2. 各インスタンスの初期化
    kvq::TurboQuant turbo_q(HEAD_DIM, NUM_BITS);
    kvq::RotorQuant rotor_q(HEAD_DIM, NUM_BITS);
    kvq::UltraQuant ultra_q(HEAD_DIM);
    kvq::LatticeQuant lattice_q(HEAD_DIM, NUM_BITS);
    kvq::RoPEAwareTurboQuant rope_tq(HEAD_DIM, 10000.0f);

    // 結果格納用テーブル
    std::cout << std::left 
              << std::setw(20) << "Algorithm"
              << std::setw(15) << "Enc Time(ms)"
              << std::setw(15) << "Dec Time(ms)"
              << std::setw(15) << "Cos Sim"
              << std::setw(15) << "MAE" << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    // ベンチマーク計測用ラムダ式
    auto run_bench = [&](const std::string& name, auto compress_fn, auto decompress_fn) {
        float total_cos_sim = 0.0f;
        float total_mae = 0.0f;

        // Encode 時間計測
        auto start_enc = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<float>> decompressed_dataset(NUM_VECTORS);

        for (size_t i = 0; i < NUM_VECTORS; ++i) {
            float scale = 0.0f;
            auto compressed = compress_fn(dataset[i], scale);
        }
        auto end_enc = std::chrono::high_resolution_clock::now();

        // 精度と Decode 時間の計測
        auto start_dec = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < NUM_VECTORS; ++i) {
            float scale = 0.0f;
            auto compressed = compress_fn(dataset[i], scale);
            decompressed_dataset[i] = decompress_fn(compressed, scale);
        }
        auto end_dec = std::chrono::high_resolution_clock::now();

        // 精度指標の平均値を算出
        for (size_t i = 0; i < NUM_VECTORS; ++i) {
            total_cos_sim += compute_cosine_similarity(dataset[i], decompressed_dataset[i]);
            total_mae += compute_mae(dataset[i], decompressed_dataset[i]);
        }

        double enc_time_ms = std::chrono::duration<double, std::milli>(end_enc - start_enc).count();
        double dec_time_ms = std::chrono::duration<double, std::milli>(end_dec - start_dec).count();

        std::cout << std::left 
                  << std::setw(20) << name
                  << std::setw(15) << std::fixed << std::setprecision(2) << enc_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(2) << dec_time_ms
                  << std::setw(15) << std::fixed << std::setprecision(4) << (total_cos_sim / NUM_VECTORS)
                  << std::setw(15) << std::fixed << std::setprecision(4) << (total_mae / NUM_VECTORS) << "\n";
    };

    // 3. 各アルゴリズムの実行

    // TurboQuant
    run_bench("TurboQuant",
        [&](const std::vector<float>& v, float& s) { return turbo_q.compress(v, s); },
        [&](const auto& q, float s) { return turbo_q.decompress(q, s); }
    );

    // RotorQuant
    run_bench("RotorQuant",
        [&](const std::vector<float>& v, float& s) { return rotor_q.compress(v, s); },
        [&](const auto& q, float s) { return rotor_q.decompress(q, s); }
    );

    // UltraQuant (FP4 + WHT)
    run_bench("UltraQuant (FP4)",
        [&](const std::vector<float>& v, float& s) { return ultra_q.compress(v, s); },
        [&](const auto& q, float s) { return ultra_q.decompress(q, s); }
    );

    // LatticeQuant (D4 Lattice)
    run_bench("LatticeQuant (D4)",
        [&](const std::vector<float>& v, float& s) { return lattice_q.compress(v, s); },
        [&](const auto& q, float s) { return lattice_q.decompress(q, s); }
    );

    // RoPE-Aware TurboQuant
    run_bench("RoPE-TurboQuant",
        [&](const std::vector<float>& v, float& s) { return rope_tq.compress(v, 0, s); },
        [&](const auto& q, float s) { return rope_tq.decompress(q, 0, s); }
    );

    std::cout << "------------------------------------------------------------------------\n";
    std::cout << "Benchmark successfully finished!\n";

    return 0;
}