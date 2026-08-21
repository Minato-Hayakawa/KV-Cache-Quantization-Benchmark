# kvq-bench 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

**English** | [日本語](README.ja.md)

A lightweight, standalone C++ micro-benchmark framework for **KV-Cache Quantization Techniques** in Large Language Models (LLMs).

This project compares state-of-the-art rotational, rate-distortion-optimal, and hardware-native quantization methods drawn from recent 2026 literature: **TurboQuant**, **RotorQuant**, **HyperQuant**, and **UltraQuant**.

---

## 🌟 Methods Compared

| Method | Key Concept | Rotation Overhead | Target Bit-width |
| :--- | :--- | :---: | :---: |
| **TurboQuant** | Cartesian → polar decomposition (PolarQuant) + QJL 1-bit sign correction | High ($O(d^2)$) | 3-bit / 8-bit |
| **RotorQuant** | Clifford-algebra rotors (Cl(3,0)) replacing full orthogonal rotation | Low ($O(d)$) | 3-bit / 8-bit |
| **HyperQuant** | Unified rate-distortion-optimal pipeline; RHT + lattice quantization ($A_2$/$D_4$/$E_8$) + Rice entropy coding | Medium | 1.7–2 bps |
| **UltraQuant** | Walsh–Hadamard rotation, QJL removed, native FP4 (E2M1) + UE8M0 block scaling for hardware-direct decode | Low | 4-bit |

---

## 📚 Related Work

### TurboQuant (Google Research, ICLR 2026)
A two-stage compression scheme:
- **PolarQuant** — converts vectors from Cartesian to polar coordinates, separating magnitude (radius) from direction (angle). Because the angular distribution is predictable, the usual per-block normalization step can be skipped.
- **QJL correction** — a Johnson–Lindenstrauss transform reduces dimensionality while preserving distances, then reduces each value to a single sign bit (+1/−1).

| Metric | Value |
| :--- | :--- |
| Compression | 6× memory reduction at 3-bit |
| Speedup | Up to 8× faster attention (H100) |
| Accuracy loss | ~Zero (validated on LongBench, RULER, etc.) |
| Models validated | Gemma, Mistral |

Reference: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)

### RotorQuant (Scrya)
Re-implements TurboQuant's core rotation using **Clifford algebra** (geometric algebra) rotors instead of a full d×d orthogonal random rotation matrix. Vectors are split into 3D groups, and each group is transformed with a 4-parameter rotor via the sandwich product $R v \tilde{R}$.

- Multiply–adds: 16,384 → ~2,064 (7.9× reduction, d=128)
- Parameters: 16,399 → 372 (44× reduction)

| Metric | Value |
| :--- | :--- |
| Speed vs. TurboQuant | 10–19× (CUDA), 9–31× (Metal) |
| Parameter count | 44× fewer (372 vs. 16,399, d=128) |
| Attention fidelity | Cosine similarity 0.990 (TurboQuant: 0.991) |
| Triton kernel | 100–650× faster than PyTorch (quant/dequant) |
| Model validated | Qwen2.5-3B-Instruct |

Reference: [github.com/scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)

### HyperQuant
A unified quantization pipeline that goes beyond localized, KV-cache-only techniques like TurboQuant, compressing **both model weights and the KV cache** under a single rate-distortion-optimal framework, reaching 1.7–2 bits/parameter (bps) with minimal accuracy loss.

- **Lattice (not scalar) quantization**: instead of rounding each dimension independently, HyperQuant uses 2D/4D/8D lattice structures ($A_2$, $D_4$, $E_8$) that pack points more densely in high-dimensional space, reducing distortion at equal bit-width.
- **Entropy coding**: a random Hadamard transform (RHT) reshapes the distribution into a near-Gaussian form; bits are then stripped using the lattice's geometric constraints and encoded with variable-length Rice coding — landing within ~0.01 bps of the theoretical rate-distortion limit.
- **Scope**: applies to LLM weights, KV caches, and even Diffusion Transformers (e.g. a 19B-parameter text-to-video model, LTX-2) — not just KV caches.

Reference: [arXiv:2606.23406](https://arxiv.org/abs/2606.23406)

### UltraQuant: 4-bit KV Caching for Context-Heavy Agents
Targets the systems/hardware bottleneck left by algorithmically elegant methods like TurboQuant: their lookup-table codebook decode and 1-bit QJL residual correction are expensive as *software* loops on GPU, hurting decode throughput (e.g. time-to-first-token) in long, multi-turn agent workloads.

- **Walsh–Hadamard rotation** spreads Key-vector outliers into a near-Gaussian distribution (as in TurboQuant).
- **QJL removal**: the 1-bit residual correction is dropped entirely to eliminate its software decode overhead.
- **Native FP4 (E2M1)**: values map directly onto hardware-supported FP4 grid points (1 sign bit, 2 exponent bits, 1 mantissa bit) instead of a custom codebook.
- **UE8M0 block scaling**: a single offline-tuned constant ($c \approx 0.156$) scales each 32-channel block, letting FP4 data feed straight into GPU matrix units (MFMA / Tensor Core instructions) with no software lookup.

Reference: [arXiv:2606.20474](https://arxiv.org/abs/2606.20474)

---

## 🧪 Experimental Setup

### Compute resources
| Platform | Vendor |
| :--- | :--- |
| ai-l40s | NVIDIA (L40S) |

### Models evaluated
- Meta-Llama 3.1 8B
- Qwen 2.5 7B
- Mistral 7B v0.3

### What we measure
- **Per-hardware execution speed & speedup** — prefill time and token generation speed (tokens/sec) at an 8K context length.
  → `eval_pt/eval_speed.py`
- **Compression ratio** at an 8K context length, against an FP16 baseline of 289 MB, measured at 2-bit, 4-bit, and 8-bit — the actual KV-cache memory footprint (in bytes) is extracted and compared to the FP16 baseline.
  → `eval_pt/eval_compression.py`
- **Accuracy degradation**, evaluated on:
  - **LongBench** — long-document summarization, QA, and code completion across many tasks → `eval_pt/eval_longbench.py`
  - **Needle in a Haystack (NIAH)** — retrieval accuracy for specific facts buried in long context → `eval_pt/eval_niah.py`
- **Attention fidelity** — cosine similarity and Top-1/Top-5 token match rate, comparing quantized attention output directly against the original FP16 output.
  → `eval_pt/eval_fidelity.py`
- **Perplexity** — how natural/coherent generated text remains after quantization.
  → `eval_pt/eval_ppl.py`

---

## 📊 Benchmark Metrics (micro-benchmark)

In addition to the model-level evaluations above, the C++ micro-benchmark itself reports:

1. **Encode & Decode Latency (µs)** — measured per key-vector using high-resolution hardware timers.
2. **Cosine Similarity** — directional accuracy of reconstructed KV vectors against the FP32 baseline.
3. **Attention Logit MAE** — mean absolute error on actual $Q \cdot K^T$ attention scores.

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
├── core_cpp/                  # 1. C++ low-level micro-benchmark
│   ├── include/
│   │   ├── turbo_quant.hpp
│   │   ├── rotor_quant.hpp    # Rotor (Clifford algebra)
│   │   ├── ultra_quant.hpp    # FP4 (E2M1) + WHT
│   │   └── lattice_quant.hpp  # HyperQuant (E8/D4 lattice)
│   └── bench_main.cpp
│
├── core_pt/                   # 2. PyTorch / CUDA custom kernels & cache
│   ├── quantizers/
│   │   ├── base.py
│   │   ├── turbo_quant.py
│   │   ├── hyper_quant.py     # RHT + lattice + Rice coding
│   │   └── ultra_quant.py     # WHT + FP4 direct mapping
│   ├── custom_cache.py        # DynamicCache interface
│   └── kernels/                # (optional) Triton/CUDA kernels
│
├── eval_pt/                    # 3. Experiment & automated evaluation scripts
│   ├── eval_speed.py           # Prefill time & tokens/sec at 8K context
│   ├── eval_compression.py     # KV-cache memory footprint vs. FP16 baseline
│   ├── eval_longbench.py       # LongBench evaluation
│   ├── eval_niah.py            # Needle in a Haystack
│   ├── eval_fidelity.py        # Attention fidelity (cosine, Top-1/5) vs. FP16
│   └── eval_ppl.py             # Perplexity
│
└── hpc_scripts/                # 4. HPC (Slurm) job scripts
    └── run_l40s_cluster.sbatch # Distributed evaluation on ai-l40s (NVIDIA L40S)
```

* **`core_cpp/`** — the standalone C++17 micro-benchmark (encode/decode latency, cosine similarity, attention logit MAE) referenced in the Quick Start above; header-only quantizer implementations plus `bench_main.cpp` as the driver.
* **`core_pt/`** — Python/PyTorch implementations wired into a `DynamicCache`-style interface for use inside an actual model's generation loop, including optional Triton/CUDA kernels.
* **`eval_pt/`** — model-level evaluation scripts (speed, compression, LongBench, NIAH, attention fidelity, perplexity) corresponding to the *Experimental Setup* section above.
* **`hpc_scripts/`** — job script for running the benchmark on the ai-l40s GPU cluster via `run_l40s_cluster.sbatch`.

---

## 📄 Citation & References

If you find this benchmark useful in your research, please cite the underlying papers:

* TurboQuant: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
* RotorQuant: [github.com/scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)
* HyperQuant: [arXiv:2606.23406](https://arxiv.org/abs/2606.23406) — HyperQuant: A Rate-Distortion-Optimal Quantization Pipeline (2026)
* UltraQuant: [arXiv:2606.20474](https://arxiv.org/abs/2606.20474) — UltraQuant: 4-bit KV Caching for Context-Heavy Agents (2026)

---

## 📜 License

This project is released under the [MIT License](LICENSE).
