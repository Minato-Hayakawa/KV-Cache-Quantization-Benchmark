# kvq-bench 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

**English** | [日本語](README.ja.md)

A lightweight, standalone C++ micro-benchmark framework for **KV-Cache Quantization Techniques** in Large Language Models (LLMs).

This project compares state-of-the-art rotational, dynamic bit-allocation, rate-distortion-optimal, and hardware-native quantization methods drawn from recent 2026 literature: **TurboQuant**, **RotorQuant**, **RoPE-Aware Bit Allocation**, **HyperQuant**, and **UltraQuant**.

---

## 🌟 Methods Compared

| Method | Key Concept | Rotation Overhead | Target Bit-width |
| :--- | :--- | :---: | :---: |
| **TurboQuant** | Cartesian → polar decomposition (PolarQuant) + QJL 1-bit sign correction | High ($O(d^2)$) | 3-bit / 8-bit |
| **RotorQuant** | Clifford-algebra rotors (Cl(3,0)) replacing full orthogonal rotation | Low ($O(d)$) | 3-bit / 8-bit |
| **RoPE-Aware Bit Allocation** | Greedy bit allocation across RoPE's 2D frequency blocks, driven by per-block sensitivity/energy scores | Medium | Mixed (4-bit / 2-bit) |
| **TurboQuant × RoPE-Aware** | RoPE-Aware allocator layered on top of TurboQuant as the base encoder | Medium–High | Mixed |
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

### RoPE-Aware Bit Allocation
Standard quantizers treat a Key vector (e.g. 128-dim) as one flat vector and apply a uniform bit-width to the whole thing. But once RoPE is applied, the attention term $q^\top R_\Delta k$ decomposes into 64 independent 2D rotation ("frequency") blocks — each contributing unevenly to the final attention logit.

- **Sensitivity scoring**: a short calibration pass estimates how much each RoPE frequency block affects the attention score.
- **Greedy bit allocation**: within a fixed average bit budget (e.g. 3 bits), high-energy blocks get 4–5 bits, low-energy blocks get 1–2 bits.
- **Block-wise quantization**: each block is quantized according to its assigned bit-width.

**Relationship to TurboQuant**: TurboQuant is a *compressor* (smooths outliers via rotation + residual correction, uniform bit-width across the whole vector, no data dependency). RoPE-Aware is an *allocator* (decides where to spend bits based on RoPE's 2D frequency structure, requires lightweight calibration). The two are complementary — RoPE-Aware is typically layered on top of TurboQuant, and especially helps on long-context and reasoning tasks where a uniform bit-width would otherwise collapse accuracy at the same memory footprint.

Reference: [arXiv:2606.24033](https://arxiv.org/abs/2606.24033)

### HyperQuant
A unified quantization pipeline that compresses **both model weights and the KV cache** under a rate-distortion-optimal framework, reaching 1.7–2 bits/parameter (bps) with minimal accuracy loss.

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
| B300 | NVIDIA |
| fs-mi300x | AMD (CDNA cluster) |
| qc-pvc | Intel (Ponte Vecchio) |
| fx700 | Fujitsu |
| Fugaku | RIKEN / Fujitsu |

### What we measure
- **Per-hardware execution speed** for each method above.
- **Compression ratio** at an 8K context length, against an FP16 baseline of 289 MB, measured at 2-bit, 4-bit, and 8-bit.
- **Speedup** relative to the FP16 baseline.
- **Accuracy degradation**, evaluated on:
  - **LongBench** — long-document summarization, QA, and code completion across many tasks
  - **Needle in a Haystack (NIAH)** — retrieval accuracy for specific facts buried in long context
  - **ZeroSCROLLS** — zero-shot long-document understanding
  - **RULER** — synthetic tasks probing context-length scaling
  - **L-Eval** — an aggregated long-context evaluation suite
- **Attention fidelity** at 3-bit (36 layers, 72 KV heads):
  - Cosine similarity: **0.9945 – 0.9961**
  - Top-1 token match rate
  - Top-5 token match rate

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
│   │   ├── rope_aware_tq.py   # TurboQuant × RoPE-Aware (hybrid)
│   │   ├── hyper_quant.py     # RHT + lattice + Rice coding
│   │   └── ultra_quant.py     # WHT + FP4 direct mapping
│   ├── custom_cache.py        # DynamicCache interface
│   └── kernels/                # (optional) Triton/CUDA kernels
│
├── eval/                       # 3. Experiment & automated evaluation scripts
│   ├── eval_ppl.py             # WikiText-2 / C4 perplexity
│   ├── eval_longbench.py       # LongBench evaluation
│   ├── eval_niah.py            # Needle in a Haystack
│   └── eval_fidelity.py        # Attention fidelity (cosine, Top-1/5)
│
└── hpc_scripts/                # 4. HPC (Slurm / Fugaku) job scripts
    ├── run_fugaku.sh           # A64FX / Fugaku CPU benchmark
    └── run_gpu_cluster.sbatch  # Distributed evaluation on H100 / GH200 / CDNA4
```

* **`core_cpp/`** — the standalone C++17 micro-benchmark (encode/decode latency, cosine similarity, attention logit MAE) referenced in the Quick Start above; header-only quantizer implementations plus `bench_main.cpp` as the driver.
* **`core_pt/`** — Python/PyTorch implementations wired into a `DynamicCache`-style interface for use inside an actual model's generation loop, including optional Triton/CUDA kernels.
* **`eval/`** — model-level evaluation scripts (perplexity, LongBench, NIAH, and attention fidelity) corresponding to the *Experimental Setup* section above.
* **`hpc_scripts/`** — job scripts for running the benchmark across the compute resources listed above (Fugaku/A64FX via `run_fugaku.sh`; GPU clusters such as B300/CDNA4 via `run_gpu_cluster.sbatch`).

---

## 📄 Citation & References

If you find this benchmark useful in your research, please cite the underlying papers:

* TurboQuant: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874)
* RotorQuant: [github.com/scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)
* RoPE-Aware Bit Allocation: [arXiv:2606.24033](https://arxiv.org/abs/2606.24033) — RoPE-Aware Bit Allocation for KV-Cache Quantization (2026)
* HyperQuant: [arXiv:2606.23406](https://arxiv.org/abs/2606.23406) — HyperQuant: A Rate-Distortion-Optimal Quantization Pipeline (2026)
* UltraQuant: [arXiv:2606.20474](https://arxiv.org/abs/2606.20474) — UltraQuant: 4-bit KV Caching for Context-Heavy Agents (2026)

---

## 📜 License

This project is released under the [MIT License](LICENSE).
