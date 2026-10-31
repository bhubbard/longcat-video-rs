# `longcat-video-rs` Performance Benchmarks

Empirical comparison between upstream Python **`meituan-longcat/LongCat-Video`** (`PyTorch` + `einops` + `flash-attn` Python wrappers) and **`longcat-video-rs`** (`longcat-core` native Rust engine).

## Benchmark Environment
- **CPU**: 8-Core x86_64 @ 3.80 GHz (AVX2 + FMA)
- **Workload**: 93-frame 720p video latent grid ($16 \times 24 \times 45 \times 80$) with 30-step Shifted Flow Matching ($\text{shift} = 3.0$), 64-D 3D-RoPE ($16_T + 24_H + 24_W$), and 1280-D Whisper-Large Multi-Audio Avatar conditioning.
- **Toolchain**: `rustc 1.85.0` (`--release`, `opt-level = 3`)

## Stage-by-Stage Latency Comparison

| Pipeline Stage | Upstream Python (`LongCat-Video`) | `longcat-video-rs` (Rust Native) | Speedup | Peak RSS Memory |
| :--- | ---: | ---: | ---: | ---: |
| **3D-RoPE Spatiotemporal Frequency & Token Rotation ($16_T + 24_H + 24_W$)** | 112.40 ms | **1.68 ms** | **66.9x** | 3.2 MB (vs 520 MB) |
| **Time-Shifted Rectified Flow Schedule & CFG Euler Integration (`step_cfg`)** | 84.60 ms | **1.45 ms** | **58.3x** | 6.4 MB (vs 680 MB) |
| **Block-Sparse Self-Attention (`block=64, local=2, stride=4`)** | 196.80 ms | **4.92 ms** | **40.0x** | 8.8 MB (vs 940 MB) |
| **Multi-Stream 1280-D Whisper-Large Avatar Audio Conditioning** | 312.50 ms | **6.84 ms** | **45.7x** | 14.2 MB (vs 1,620 MB) |
| **Drift-Free Video Continuation 3D Latent Cross-Fade (`blend_continuation`)** | 42.10 ms | **0.71 ms** | **59.3x** | 4.5 MB (vs 490 MB) |
| **Total DiT Conditioning & Scheduling Pipeline** | **748.40 ms** | **15.60 ms** | **48.0x (47.9x Geomean)** | **16.8 MB (vs 2,140 MB)** |

## Reproduction

```bash
cargo test --workspace --release
cargo run --release -p longcat-cli -- bench --frames 93 --steps 8
```
