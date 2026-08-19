# kvq-bench 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

A lightweight, standalone C++ micro-benchmark framework for **KV-Cache Quantization Techniques** in Large Language Models (LLMs).

This project compares state-of-the-art rotational, dynamic bit-allocation, and ultra-lightweight quantization methods—inspired by recent 2026 arXiv literature (e.g., *RoPE-Aware Bit Allocation [2606.24033]*, *TurboQuant*, and *RotorQuant*).

---

## 🌟 Implemented Methods

| Method | Key Concept | Rotation Overhead | Target Bit-width |
| :--- | :--- | :---: | :---: |
| **TurboQuant** | Full orthogonal random matrix rotation + uniform quantization | High ($O(d^2)$) | 3-bit / 8-bit |
| **RotorQuant** | Block-wise 2D/4D local rotations (Clifford/Givens-like) | Low ($O(d)$) | 3-bit / 8-bit |
| **RoPE-Aware TQ** | Dynamic bit-allocation based on RoPE positional frequencies | Medium | Mixed (4-bit/2-bit) |
| **BitNetSimplified** | No rotation; ternary sign-based quantization ($[-1, 0, 1]$) | Zero | 1.58-bit |

---

## 📊 Benchmark Metrics

We evaluate each algorithm across three main axes:

1. **Encode & Decode Latency (µs):** Measured per key-vector using high-resolution hardware timers.
2. **Cosine Similarity:** Directional accuracy of the reconstructed KV vectors against FP32 baselines.
3. **Attention Logit MAE:** Mean Absolute Error on actual $Q \cdot K^T$ attention scores.

---

## 🚀 Quick Start

### Prerequisites

* C++17 compliant compiler (`g++` or `clang++`)
* CMake 3.14+
* (Optional) Python 3.8+ with `pandas` and `matplotlib` for plotting

### Building and Running

```bash
# 1. Clone the repository
git clone https://github.com/your-username/kvq-bench.git
cd kvq-bench

# 2. Build via CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. Run the micro-benchmark
./build/bench_kv
```

Running `./build/bench_kv` will output real-time execution stats in terminal and generate a `results.csv` file.

### Plotting Results

```bash
python3 scripts/plot_results.py
```

This generates `benchmark_result.png` displaying latency vs. similarity trade-offs.

---

## 📁 Repository Structure

```
kvq-bench/
├── include/kvq/       # Modular header files (.hpp) for each quantizer
├── src/               # Implementation files (.cpp)
├── bench/             # Main benchmark loop & CSV exporter (main_bench.cpp)
├── scripts/           # Python plotting utility & Slurm job scripts
├── CMakeLists.txt     # Build configuration
└── README.md
```

---

## 📄 Citation & References

If you find this benchmark useful in your research, please cite the underlying papers:

* RoPE-Aware Bit Allocation: [2606.24033] RoPE-Aware Bit Allocation for KV-Cache Quantization (2026)
* HyperQuant: [2606.23406] HyperQuant: A Rate-Distortion-Optimal Quantization Pipeline (2026)
* UltraQuant: [2606.20474] UltraQuant: 4-bit KV Caching for Context-Heavy Agents (2026)
* TurboQuant & RotorQuant

---

## 📜 License

This project is released under the [MIT License](LICENSE).
