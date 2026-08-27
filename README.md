# kvq-bench 🚀

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

**English** | [日本語](README.ja.md)

A lightweight benchmark framework for **KV-Cache Quantization Techniques** in Large Language Models (LLMs).

It contains simplified PyTorch re-implementations **inspired by** four recent methods — **TurboQuant**, **RotorQuant**, **HyperQuant**, and **UltraQuant** — plus a standalone C++ micro-benchmark kernel suite.

> ⚠️ **Scope disclaimer**: the algorithms here are *simplified re-implementations*, not faithful reproductions of the papers (see [Scope & Limitations](#-scope--limitations)). Numbers in *Related Work* are **from the original papers**, not this repository's results.

---

## 🌟 Methods (simplified re-implementations)

| Method | Paper's idea | This repo implements | Paper's claimed bit-width | This impl's `designed_bits` |
| :--- | :--- | :--- | :---: | :---: |
| **TurboQuant** | Polar decomposition + QJL | Random orthogonal rotation + per-token abs-max uniform quantization | 3-bit / 8-bit | 3.0 |
| **RotorQuant** | Clifford-algebra rotors | Cl(3,0) 3D-block rotor rotation + per-3D-block scaling (random fixed rotors, no calibration) | 3-bit / 8-bit | 3.0 |
| **HyperQuant** | RHT + lattice + Rice coding | Orthogonal Hadamard + **D4 lattice projection** (Rice entropy coding **not** implemented) | 1.7–2 bps | 3.0 |
| **UltraQuant** | FP4 hardware-direct decode | Walsh–Hadamard + FP4 (E2M1) grid mapping, per-token scale | 4-bit | 4.0 |

`designed_bits` is **derived from each implementation's quantization levels** (auto-computed, not hand-set). Sub-4-bit claims *as rates* (bps) require entropy coding, which is out of scope here — so despite the HyperQuant paper operating at 1.7–2 bps, this implementation can only achieve 3 bit/scalar.

---

## 🔭 Scope & Limitations

What this benchmark **can** claim (after the 2026-08 fixes):

- **Functional / simulated-quantization quality evaluation** — quantize→dequantize math is applied as designed, and quality metrics (PPL, logit/KV fidelity, NIAH success rate, LongBench QA-F1) reflect the implementations' actual precision loss.
- **Throughput of the quantization layer itself** — i.e., the overhead the quantizer adds on top of fp16 attention.

What it **cannot** claim (by design):

- **Decode speedups from quantization** — attention always runs on dequantized full-precision tensors; no fused kernel means no memory-bandwidth benefit. Speed numbers are overhead measurements only.
- **Measured memory savings matching designed bit-widths** — stored tensors are int8 (no bit-packing), so actual stored bytes do not reflect `designed_bits`. Memory is reported as an **analytical footprint** (data at designed bits + modeled metadata), not a VRAM measurement.
- **Paper-faithful reproduction** — PolarQuant/QJL (TurboQuant), rotor calibration (RotorQuant), Rice coding (HyperQuant), UE8M0 block scaling & MFMA integration (UltraQuant) are not implemented.
- **Pre-RoPE (KIVI-style) quantization** — this cache receives post-RoPE keys; pre-RoPE schemes require model-surgery outside this framework.
- **Strong statistical claims** — small sample sizes by design; results are directional observations.

---

## 📚 Related Work (paper numbers — independent of this repo)

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

**Independent validation**: A comprehensive third-party study by the vLLM team (May 2026) benchmarked TurboQuant on production-scale models (Llama-3.3-70B-Instruct, Qwen3-30B-A3B, MiniMax-M2.7) across long-context retrieval (MRCR) and reasoning benchmarks (AIME25, GPQA, MATH500, LiveCodeBench-v6). The results were more mixed than the original paper: FP8 KV-cache quantization matched BF16 throughput at 2× capacity with negligible accuracy loss, while TurboQuant's `k8v4` variant offered only marginal gains over FP8 at a 40–52% throughput cost. The `4bit-nc` variant was judged the most practical TurboQuant option — useful under memory pressure — while the more aggressive `k3v4-nc` / `3bit-nc` variants showed meaningful accuracy drops (up to ~20 points) on hard reasoning and coding tasks, making FP8 KV-cache the recommended default in production.

Reference: [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) · [vLLM Blog — A First Comprehensive Study of TurboQuant](https://vllm.ai/blog/2026-05-11-turboquant)

### RotorQuant (Scrya)
Re-implements TurboQuant's core rotation using **Clifford algebra** (geometric algebra) rotors instead of a full d×d orthogonal random rotation matrix. Vectors are split into 3D groups, and each group is transformed with a 4-parameter rotor via the sandwich product $R v \tilde{R}$.

- Multiply–adds: 16,384 → ~2,064 (7.9× reduction, d=128, per the RotorQuant paper's Table 1)
- Parameters: 16,399 → 372 (44× reduction, per the RotorQuant paper's Table 1)

| Metric | Value |
| :--- | :--- |
| Perplexity (WikiText-2, Llama 3.1 8B Instruct, 10.3× compression) | 6.91 (`iso3`) vs. 7.07 (TurboQuant `turbo3`) — better quality at equal compression |
| Decode speed vs. TurboQuant | 28% faster (119 vs. 93 tok/s, RTX 5090) |
| Prefill speed vs. TurboQuant | 5.3× faster (3,822 vs. 722 tok/s, RTX 5090) |
| Parameter count | 44× fewer (372 vs. 16,399, per the RotorQuant paper's Table 1) |
| Models validated | Llama 3.1 8B Instruct (headline benchmarks); Qwen2.5-3B (decode-speed and Python/Triton perplexity benchmarks); MiniMax-M2.7 (architecture-compatibility check only, not a full benchmark) |

**Important nuance**: the repository's headline "beats TurboQuant" numbers above actually come from two simpler derivative methods built on the same block-diagonal-rotation idea — **IsoQuant** (4D quaternion rotation) and **PlanarQuant** (2D Givens rotation), both credited to a separate contributor (ParaMind2025) — rather than from the original Clifford-algebra RotorQuant method itself. In the repository's own Python/Triton perplexity comparison on Qwen2.5-3B, plain RotorQuant (PPL 12.22 at 3-bit, 10.03 at 4-bit) actually performs *worse* than both IsoQuant (9.03 at 4-bit) and PlanarQuant (10.12 at 3-bit). The repository labels RotorQuant itself as "Research (Triton)" status, while IsoQuant/PlanarQuant are the "Production (llama.cpp)" variants. Earlier claims of "10–19× CUDA / 9–31× Metal speedups" and a specific cosine-similarity figure for RotorQuant could not be verified against the current repository content and have been removed.

Reference: [github.com/scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)

### HyperQuant
A unified quantization pipeline that goes beyond localized, KV-cache-only techniques like TurboQuant, compressing **both model weights and the KV cache** under a single rate-distortion-optimal framework, reaching 1.7–2 bits/parameter (bps) with minimal accuracy loss.

- **Lattice (not scalar) quantization**: instead of rounding each dimension independently, HyperQuant uses 2D/4D/8D lattice structures ($A_2$, $D_4$, $E_8$) that pack points more densely in high-dimensional space, reducing distortion at equal bit-width.
- **Entropy coding**: a random Hadamard transform (RHT) reshapes the distribution into a near-Gaussian form; bits are then stripped using the lattice's geometric constraints and encoded with variable-length Rice coding — landing within ~0.01 bps of the theoretical rate-distortion limit.
- **Scope**: applies to LLM weights, KV caches, and even Diffusion Transformers (e.g. a 19B-parameter text-to-video model, LTX-2) — not just KV caches.

| Metric | Value |
| :--- | :--- |
| vs. TurboQuant / OCTOPUS (KV path) | Outperforms both down to 1.7 bits/scalar (bps) |
| vs. HIGGS (weight path) | Outperforms at every operating point from 3–5 bps |
| Compression (H100, 4 bps, near-lossless) | ~3.9× weights, ~3.79× KV cache |
| Rate-distortion gap | Within ~0.01 bps of the theoretical limit |
| Models validated | Llama-3.1-8B (KV-only path, bf16 weights); LTX-2 (19B-parameter DiT video model, no observable per-frame artifacts) |
| Triton/CUDA kernel speedup | *Not reported in the publicly available paper text* |

Reference: [arXiv:2606.23406](https://arxiv.org/abs/2606.23406)

### UltraQuant: 4-bit KV Caching for Context-Heavy Agents
Targets the systems/hardware bottleneck left by algorithmically elegant methods like TurboQuant: their lookup-table codebook decode and 1-bit QJL residual correction are expensive as *software* loops on GPU, hurting decode throughput (e.g. time-to-first-token) in long, multi-turn agent workloads.

- **Walsh–Hadamard rotation** spreads Key-vector outliers into a near-Gaussian distribution (as in TurboQuant).
- **QJL removal**: the 1-bit residual correction is dropped entirely to eliminate its software decode overhead.
- **Native FP4 (E2M1)**: values map directly onto hardware-supported FP4 grid points (1 sign bit, 2 exponent bits, 1 mantissa bit) instead of a custom codebook.
- **UE8M0 block scaling**: a single offline-tuned constant ($c = 0.156$, confirmed MSE-optimal in the paper's ablations) scales each 32-channel block, letting FP4 data feed straight into GPU matrix units (MFMA / Tensor Core instructions) with no software lookup.

| Metric | Value |
| :--- | :--- |
| Headline: TTFT vs. FP8 KV (agentic, late rounds / all rounds) | 3.47× faster / 2.3× faster |
| Headline: output throughput vs. FP8 KV (agentic) | 1.63× higher |
| Throughput vs. BF16 (standard serving, concurrency 64) | 1.38× (vs. FP8 KV's 1.37×, within ~1%), while using half the KV bytes per element of FP8 |
| Median time-per-output-token vs. BF16 | 1.40× (FP8 KV: 1.37×; Ultra-TQ: 1.58×; vLLM OSS TurboQuant: 5.56×) |
| UE8M0 scaling constant | $c = 0.156$ — confirmed MSE-optimal; beats an FP8 baseline by +4.4 pp on the paper's GPQA ablation |
| Hardware | AMD Instinct MI355X (CDNA4), TP=2, native scaled-MFMA instructions |
| Models validated | MiniMax-M2.5 (throughput/latency); production accuracy matrix: Qwen3.5-A3B, MiniMax-M2.5, Qwen2.5-72B across GPQA-Diamond, LCB-128K, AIME25, MATH500 |
| Accuracy impact (production matrix) | Stable-to-positive on MATH500 (+0.0 to +0.8 pp); competitive on GPQA-Diamond and LCB-128K; **material regression on AIME25** (−13.3 pp Qwen3.5-A3B, −10.0 pp MiniMax-M2.5, −3.3 pp Qwen2.5-72B) — the authors explicitly describe this as benchmark-dependent rather than uniformly near-lossless. All results use boundary-layer protection (first/last 2 attention layers kept in BF16) |

Reference: [arXiv:2606.20474](https://arxiv.org/abs/2606.20474)

---

## 🧪 Experimental Setup

### Compute resources
| Platform | Vendor |
| :--- | :--- |
| ai-l40s | NVIDIA (L40S) |

### Models evaluated
- Meta-Llama 3.1 8B (`meta-llama/Meta-Llama-3.1-8B`)
- Mistral 7B v0.3 (`mistralai/Mistral-7B-Instruct-v0.3`)

### 0. Sanity gate (regression test) — `eval_pt/sanity_check.py`
Runs **first**, and the whole evaluation aborts if it fails.

**Common protocol**: greedily generate **16 tokens** from a fixed ~256-token prompt — not via `model.generate`, but through a manual prefill → one-token-at-a-time decode loop (the same path used by the eval scripts) — and compare the token sequence against the fp16 baseline. Model precision matches the other evaluations (bfloat16 on CUDA).

#### Shared by all methods — cache plumbing (passthrough)
The **`passthrough`** identity "quantizer" must reproduce fp16 generation **token-for-token**; it is the only criterion that enters `overall_pass`. This validates the cache plumbing (accumulation, concatenation, full-history return) in isolation from quantization error. The historical decode bug would have been caught here immediately.

#### turbo_quant / rotor_quant / hyper_quant — 7-bit round-trip test
For the three methods whose bit-width is a continuous precision knob, run each at **7-bit** (higher than production) and report the token match rate vs fp16 (advisory criterion: ≥ 0.5 counts as near-lossless; informational only and not part of `overall_pass`). Because higher bits run the *same* round-trip code path with only finer rounding, near-lossless generation at 7-bit validates the quantizer round-trip math.

#### ultra_quant — dedicated sanity experiment (separate path)
ultra_quant is defined by a **fixed 16-point FP4 (E2M1) grid** — a "7-bit equivalent" mode does not exist — so it is excluded from the 7-bit loop. Instead, a dedicated script `eval_pt/sanity_check_ultra.py` (sbatch: `hpc_scripts/run_sanity_ultra_ai_l40s.sh`) runs two checks (implemented as in-script subclasses; no changes to existing code):

1. **Lossless-decomposition test** (plumbing + rotation/scale math): a high-precision bypass variant — identical WHT rotation and absmax/6 scale round-trip with *only* the FP4 grid snap removed (storage effectively fp32) — must match fp16 tokens at a rate **≥ 0.9** (expected ≈ 1.0; a large miss indicates broken rotation/scale plumbing).
2. **Native 4-bit grid integrity** (model-free numeric check): quantize→dequantize a seeded random tensor in the production 4-bit mode and assert (a) relative L2 error < 0.40 and (b) every stored index lies in 0–15 (catches int8-overflow-style bugs). The grid mapping itself cannot be judged by a near-losslessness argument, so it is verified with an explicit error bound instead.

### Quality metrics (functional / simulated quantization)
- **Perplexity** — WikiText-2, window 2048 / stride 512 → `eval_pt/eval_ppl.py`
- **Logit fidelity** — cosine similarity, Top-1 match, Top-5 **overlap** rate of *final logits* against the fp16 run (seq 1024, natural non-repetitive text) → `eval_pt/eval_fidelity.py`
  - Note: with repetitive synthetic input, next-token margins are huge at almost every position, so Top-1 **saturates** (e.g., pinned at 1023/1024 ≈ 99.90% for all conditions) and loses discriminative power. The Top-1 column in the table below is an old measurement under that saturated regime, shown for reference only.
- **KV fidelity** — cosine similarity and relative L2 error of the cached K/V vectors themselves vs fp16 (isolates quantization error from model nonlinearity) → same script
- **NIAH (Needle In A Haystack)** — success **rate** over depths {10/30/50/70/90%} × 5 trials per depth (= 25 trials) at 8K context, greedy 16-token decode, exact key match → `eval_pt/eval_niah.py`
- **LongBench** — real LongBench **Qasper** QA task, first 10 samples, official-style token F1, context truncated to 6144 from the left → `eval_pt/eval_longbench.py`

### Systems metrics (carefully framed)
- **Analytical KV footprint** — per-method `designed_bits` (derived from the implementation), data bytes at that bit-width, plus implementation-faithful metadata (per-token / per-3D-block scales stored as fp32) → `eval_pt/theoretical_compression.py` (GPU-free) and `eval_pt/eval_compression.py` (adds actual stored bytes of the int8 simulation).
- **Speed** — prefill time and decode tokens/sec, median of 3 runs after 1 warmup, 8K context, 32 generated tokens. **Interpretation: quantization-overhead-inclusive speed only; no speedup claims.** → `eval_pt/eval_speed.py`

---

## 📊 Benchmark Metrics (C++ micro-benchmark)

The standalone C++ micro-benchmark reports:

1. **Encode & Decode Latency (µs)** — measured per key-vector using high-resolution hardware timers.
2. **Cosine Similarity** — directional accuracy of reconstructed KV vectors against the FP32 baseline.
3. **Attention Logit MAE** — mean absolute error on actual $Q \cdot K^T$ attention scores.

---

## 📈 Experimental Results

Models: `meta-llama/Meta-Llama-3.1-8B`, `mistralai/Mistral-7B-Instruct-v0.3`
Methods: fp16 (baseline), turbo_quant, rotor_quant, hyper_quant, ultra_quant

> 🗂️ **Provenance**: the tables below are from the 2026-08 re-measurement with the fixed harness (jobs 373124 + 373434, plus job 373457 for NIAH at `trials_per_depth=5` and the fp16 LongBench re-run, and job 373515 for the ultra_quant sanity; full machine-aggregated tables in [`results/ai-l40s/valid_results_summary.md`](results/ai-l40s/valid_results_summary.md); merged data in [`summary_results.csv`](summary_results.csv) via `python results/summarize_results.py`).

### 1. KV cache footprint — analytical (8K context, derived values)

Both models share the same KV geometry (32 layers × 8 KV heads × 128 head_dim ⇒ 1,024 MB at fp16 @ 8,192 tokens), so a single table applies to both. Bit-widths are derived from the implementations; metadata is modeled from each implementation's actual scale storage (fp32).

| Method | designed_bits | Data (MB) | Metadata (MB) | **Designed footprint (MB)** | Compression vs fp16 |
| :--- | :---: | ---: | ---: | ---: | ---: |
| fp16 | 16.0 | 1024.0 | 0.0 | 1024.0 | 1.00× |
| turbo_quant | 3.0 | 192.0 | 16.0 | 208.0 | **4.92×** |
| hyper_quant | 3.0 | 192.0 | 16.0 | 208.0 | **4.92×** |
| rotor_quant | 3.0 | 192.0 | 704.0 | 896.0 | **1.14×** |
| ultra_quant | 4.0 | 256.0 | 16.0 | 272.0 | **3.76×** |

- hyper_quant shows the same designed footprint as turbo_quant because **Rice entropy coding is not implemented** — without it, the paper's 1.7–2 bps rates are unreachable here.
- rotor_quant's **per-3D-block fp32 scales** make its metadata (704 MB) larger than its 3-bit data (192 MB) — an implementation-structure insight: fine-grained scaling destroys the headline compression.

For reference, the simulation's *actual stored bytes* (int8 + fp32 scales, no packing — measured deterministically on the older runs, and equally valid for the fixed code) are: fp16 1,024 MB; turbo/hyper/ultra 528 MB; rotor 1,208 MB. Identical across bit-widths **by construction** (everything is stored as int8), which is exactly why bit-width claims must come from the analytical table above, not from stored bytes.

![Compression comparison](plots/compression_comparison.png)

### 2. Quality — PPL, logit / KV fidelity (re-measured with the fixed harness)

Values below were re-measured with the 2026-08 fixed harness. They reflect the seed-fixed quantizers and HyperQuant's now-actually-applied D4 lattice projection, so they differ from the previously posted pre-fix numbers.

**Perplexity (WikiText-2, window 2048 / stride 512 — lower is better)**

| Model | Method | PPL |
| :--- | :--- | ---: |
| Meta-Llama-3.1-8B | fp16 | 5.5667 |
| | turbo_quant | 6.8175 |
| | rotor_quant | 5.7484 |
| | hyper_quant | 7.3255 |
| | ultra_quant | 5.6922 |
| Mistral-7B-Instruct-v0.3 | fp16 | 4.8756 |
| | turbo_quant | 5.2196 |
| | rotor_quant | 4.9276 |
| | hyper_quant | 5.3571 |
| | ultra_quant | 4.9072 |

![Perplexity comparison](plots/ppl_comparison.png)

**logit / KV fidelity (seq 1024)**

These are single-forward (prefill-style) evaluations. Fidelity is measured in **final-logit space** (not attention outputs), Top-5 is an **overlap rate**, and KV fidelity is the error of the cached K/V vectors themselves (the quantization error proper).

> **Note on the Top-1 column**: the table was produced by the old script (repetitive synthetic text), where Top-1 saturates at 99.5–99.90% for all conditions (1023/1024 ≈ 99.90% is the lattice point meaning "argmax flipped at exactly 1 of 1024 positions", and flips concentrate at low-margin positions near BOS). The script has since been fixed to use natural non-repetitive text, and this column will be re-measured. The other columns (logit cos, Top-5, KV error) are unaffected by the saturation and remain valid.

| Model | Method | Logit cos | Logit Top-1 (%) | Logit Top-5 overlap (%) | KV cos | KV rel. L2 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Meta-Llama-3.1-8B | fp16 | 1.000000 | 100.00 | 100.00 | 1.000000 | 0.0000 |
| | turbo_quant | 0.707082 | 99.90 | 49.53 | 0.775749 | 0.6176 |
| | rotor_quant | 0.975850 | 99.80 | 84.04 | 0.947499 | 0.2886 |
| | hyper_quant | 0.685053 | 99.71 | 40.78 | 0.753620 | 0.6562 |
| | ultra_quant | 0.970683 | 99.90 | 85.55 | 0.940746 | 0.3033 |
| Mistral-7B-Instruct-v0.3 | fp16 | 1.000000 | 100.00 | 100.00 | 1.000000 | 0.0000 |
| | turbo_quant | 0.985607 | 99.90 | 66.29 | 0.870971 | 0.4841 |
| | rotor_quant | 0.999409 | 99.90 | 94.47 | 0.983329 | 0.1717 |
| | hyper_quant | 0.977383 | 99.51 | 59.88 | 0.845100 | 0.5343 |
| | ultra_quant | 0.999443 | 99.90 | 94.16 | 0.981794 | 0.1759 |

![Fidelity comparison](plots/fidelity_comparison.png)

> Observation: on Mistral-7B every method keeps high logit cosine (0.977–0.999), but turbo/hyper still drop to 59.9–66.3% Top-5 overlap (rotor/ultra stay ≈94%). On Llama-3.1-8B the 3-bit per-token-scaled methods (turbo/hyper) degrade much more (0.685–0.707 cos, KV rel. L2 up to 0.62–0.66). rotor/ultra stay high — consistent with their finer scaling (per-3D-block / 4-bit grid). Susceptibility clearly depends on the base model.

### 3. Re-measured results (NIAH, LongBench, speed)

**Sanity gate**: PASS on both models (passthrough reproduces fp16 token-for-token; 7-bit quantizers near-lossless — Llama's hyper_quant 7-bit matched at 0.875, above the 0.5 criterion; ultra_quant is not part of the 7-bit gate and is covered separately via `run_sanity_ultra_ai_l40s.sh`).

**NIAH (8K context, success rate over 5 depths × 5 trials = 25 trials)**

| Model | Method | Success rate |
| :--- | :--- | :---: |
| Meta-Llama-3.1-8B | fp16 | 25/25 (1.00) |
| | turbo_quant | 25/25 (1.00) |
| | rotor_quant | 25/25 (1.00) |
| | hyper_quant | 25/25 (1.00) |
| | ultra_quant | 25/25 (1.00) |
| Mistral-7B-Instruct-v0.3 | fp16 | 25/25 (1.00) |
| | turbo_quant | 16/25 (0.64) |
| | rotor_quant | 19/25 (0.76) |
| | hyper_quant | 5/25 (0.20) |
| | ultra_quant | 25/25 (1.00) |

Re-measured at 5 trials per depth (job 373457) for better statistical resolution; the 25-trial protocol also reveals degradation the 5-trial version missed (Mistral turbo_quant dropped from 5/5 to 16/25). The model-dependence is inverted vs. PPL/fidelity: on retrieval, Mistral-7B suffers under the 3-bit methods (turbo 0.64 / rotor 0.76 / hyper 0.20) while Llama-3.1-8B stays perfect across all five methods; ultra_quant is retrieval-safe on both. The old "all methods False" conclusion was a harness bug and remains retracted.

![NIAH success rate](plots/niah_success_rate.png)

**LongBench (Qasper QA-F1, first 10 samples, context truncated to 6144 from the left — higher is better)**

| Model | Method | Mean QA-F1 |
| :--- | :--- | ---: |
| Meta-Llama-3.1-8B | fp16 | 0.2997 |
| | turbo_quant | 0.1580 |
| | rotor_quant | 0.2551 |
| | hyper_quant | 0.1859 |
| | ultra_quant | 0.2832 |
| Mistral-7B-Instruct-v0.3 | fp16 | 0.3256 |
| | turbo_quant | 0.3135 |
| | rotor_quant | 0.2813 |
| | hyper_quant | 0.2582 |
| | ultra_quant | 0.2856 |

The two fp16 runs were re-measured fresh in job 373457, superseding the earlier log-restored files; the mean F1 values reproduced the previously reported numbers exactly (0.2997 / 0.3256). The old results (un-scored degenerate text) were caused by the cache bug and are **retracted**.

![LongBench QA-F1](plots/longbench_f1.png)

**Speed (8K context, 32 generated tokens, median of 3 — a quantization-overhead measurement; no speedup claims)**

| Model | Method | Prefill (s) | Decode (s) | Tokens/s | Slowdown vs fp16 |
| :--- | :--- | ---: | ---: | ---: | ---: |
| Meta-Llama-3.1-8B | fp16 | 1.2504 | 0.9545 | 33.52 | 1.00× |
| | turbo_quant | 1.2895 | 1.2064 | 26.53 | 1.26× |
| | rotor_quant | 1.4197 | 2.2416 | 14.28 | 2.35× |
| | hyper_quant | 1.3413 | 1.7750 | 18.03 | 1.86× |
| | ultra_quant | 1.6308 | 1.2326 | 25.96 | 1.29× |
| Mistral-7B-Instruct-v0.3 | fp16 | 1.2066 | 0.9167 | 34.91 | 1.00× |
| | turbo_quant | 1.2473 | 1.1662 | 27.44 | 1.27× |
| | rotor_quant | 1.3771 | 2.2435 | 14.26 | 2.45× |
| | hyper_quant | 1.2968 | 1.7350 | 18.44 | 1.89× |
| | ultra_quant | 1.5876 | 1.2001 | 26.67 | 1.31× |

Every quantized method is slower than fp16 (the quantizer's added cost); rotor's per-3D-block scaling makes it the slowest (2.35–2.45×). The old speed table (incl. ultra_quant beating fp16) is **retracted** — decode comparisons were not apples-to-apples.

![Speed comparison](plots/speed_comparison.png)

---

## 🧠 Discussion

### 1. Compression must be discussed at designed bits + metadata, not stored bytes

Because the simulation stores everything as int8 + fp32 scales, all methods previously "measured" 528 MB regardless of bit-width — a tautology of the storage format, not a finding. The analytical view is the honest one: turbo/hyper ≈ 4.9×, ultra ≈ 3.8×, and rotor collapses to ≈ 1.1× because per-3D-block fp32 scales outgrow the 3-bit payload. Real hardware numbers would further require true bit-packing.

### 2. Quantization robustness differs by base model (GQA etc.)

PPL, logit/KV fidelity, and LongBench QA-F1 all consistently show Llama-3.1-8B suffering more than Mistral-7B under the 3-bit per-token methods (turbo/hyper). Even on the end task, turbo nearly halves its QA-F1 on Llama (0.1580 vs 0.2997 fp16) while almost matching fp16 on Mistral (0.3135 vs 0.3256). Architecture-dependent robustness is real, and there is no universally "safe" method among these simplified implementations. On the overall balance of quality, footprint, and overhead, ultra_quant (4-bit, 272 MB, ~1.3× slowdown) came out as the most robust option here.

### 3. What this benchmark deliberately does not claim

No decode-speedup claims (no fused kernels), no production memory claims (analytical footprint only), no paper-reproduction claims (simplified algorithms), and no strong statistical claims (small samples). The framework's value is in **functional quality comparison under simulated quantization**, which requires the cache plumbing to be exactly correct — hence the `sanity_check.py` gate.

---

## ✅ Conclusion

Re-defined goal of this benchmark: *a functional, mathematically honest comparison of simplified KV-cache quantizers in quality terms (PPL, fidelity, retrieval, QA), with memory expressed as an analytical footprint and speed recorded as quantization overhead — with no claims beyond what a kernel-free simulation can support.*

---

## ♻️ Fix History & Retractions (2026-08)

Previous results contained serious harness bugs. Documented for transparency:

1. **Cache `update()` returned only the latest chunk** (dequantized), not full history — so during decode every quantized method attended to only the newest token. → *Invalidates*: NIAH table ("all False"), LongBench outputs (degenerate text), speed table (decode was not comparable; ultra beating fp16). **These conclusions are retracted.**
2. **RotorQuant concatenation condition** matched only dicts containing a `"quantized"` key; rotor uses `"quantized_main"`, so its decode history was broken even beyond bug 1.
3. **Compression baseline 289 MB was a hardcoded constant** in an older script version, and the reported "ratio" was `289/measured` (inverted semantics) — replaced by stored-bytes + analytical footprint.
4. **`designed_bits` were hand-set** (ultra=2.0 etc.) instead of derived — now auto-derived from each quantizer's levels (ultra=4.0).
5. Additional fixes: D4 lattice now actually applied in HyperQuant's store path (it was dead code), quantizers are seed-fixed (previously non-deterministic across runs), fidelity metrics correctly labeled as logit-space, NIAH/LongBench upgraded to scored protocols.

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
├── core_pt/                   # 2. PyTorch simplified quantizers (simulated quantization)
│   ├── quantizers/
│   │   ├── __init__.py
│   │   ├── base.py            # base class (designed_bits attribute)
│   │   ├── turbo_quant.py     # rotation + uniform (simplified)
│   │   ├── rotor_quant.py     # 3D-block rotors (simplified)
│   │   ├── hyper_quant.py     # Hadamard + D4 lattice (no Rice coding)
│   │   └── ultra_quant.py     # WHT + FP4 (E2M1) grid
│   └── kernels/               # (optional) Triton/CUDA kernels
│
├── eval_pt/                    # 3. Model-level evaluation (Hugging Face)
│   ├── custom_cache.py        # QuantizedKVCache (+ passthrough for sanity)
│   ├── sanity_check.py        # ★ regression gate: aborts eval if plumbing breaks
│   ├── sanity_check_ultra.py  # ultra_quant-only sanity (separate path; no 7-bit mode)
│   ├── eval_ppl.py            # Perplexity (WikiText-2, 2048/512)
│   ├── eval_fidelity.py       # Logit + KV fidelity
│   ├── eval_niah.py           # Needle in a Haystack (multi-depth success rate)
│   ├── eval_longbench.py      # LongBench Qasper QA-F1
│   ├── eval_speed.py          # prefill/decode speed, median (overhead only)
│   ├── eval_compression.py    # stored bytes + analytical footprint
│   └── theoretical_compression.py  # GPU-free analytical footprint
│
└── hpc_scripts/                # 4. HPC (Slurm) job scripts
    ├── run_all_evals_ai_l40s.sh      # sanity → footprint → ppl → fidelity → niah → longbench → speed
    ├── run_sanity_ultra_ai_l40s.sh   # ultra_quant-only sanity
    └── run_rerun_niah_fp16longbench_ai_l40s.sh  # NIAH re-run (trials_per_depth=5) + fp16 LongBench re-run
```

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
