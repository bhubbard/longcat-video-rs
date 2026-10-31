# `longcat-video-rs` — 13.6B Unified Video Diffusion Transformer (3D-RoPE, Shifted Flow Matching & Block-Sparse Attention) in Pure Rust

[![CI](https://github.com/bhubbard/longcat-video-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/bhubbard/longcat-video-rs/actions/workflows/ci.yml)
[![Interactive Studio](https://img.shields.io/badge/Interactive_Studio-code.brandonhubbard.com%2Flongcat--video--rs-2b5cff?style=flat-square)](https://code.brandonhubbard.com/longcat-video-rs/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

**`longcat-video-rs`** is a native, memory-safe Rust port of Meituan's [**LongCat-Video & LongCat-Video-Avatar 1.5**](https://github.com/meituan-longcat/LongCat-Video) 13.6B Unified Diffusion Transformer (DiT) engine. It unifies five core video synthesis tasks — **Text-to-Video (`T2V`)**, **Image-to-Video (`I2V`)**, **Drift-Free Video Continuation (`Continue`)**, **Single-Speaker Audio Avatar (`AvatarSingleAudio`)**, and **Multi-Speaker Dialog Avatar (`AvatarMultiAudio`)** — powered by partitioned **3D Rotary Position Embeddings (`RoPE3D`: $16_T + 24_H + 24_W = 64$)**, **Time-Shifted Rectified Flow Matching (`FlowMatchingEulerScheduler`)**, and **Local + Strided Global Block-Sparse Attention**.

---

## Core Architecture (`longcat-core` + `longcat-cli`)

- **`crates/longcat-core/src/rope_3d.rs`**:
  - `RoPE3D`: Partitioned 3D spatiotemporal rotary embeddings (`temporal_dim = 16`, `height_dim = 24`, `width_dim = 24`, $\theta_{\text{base}} = 10000.0$) conserving token $L_2$ norm across arbitrary $(t, h, w)$ coordinates (`apply_rope`).
- **`crates/longcat-core/src/flow_match.rs`**:
  - `FlowMatchingEulerScheduler`: Computes resolution-shifted Rectified Flow schedules $\sigma_{\text{shifted}} = \frac{s \cdot t}{1 + (s - 1) t}$ (`get_schedule`) and executes both standard (`step`) and Classifier-Free Guided (`step_cfg`) Euler ODE steps.
- **`crates/longcat-core/src/attention.rs`**:
  - `Attention::block_sparse_self_attention`: Reduces $O(N^2)$ long-video DiT self-attention to linear-strided block sparsity (`block_size = 64`, `local_blocks = 2`, `stride_blocks = 4`).
  - `Attention::audio_cross_attention`: Cross-attends video latent tokens against 1280-dimensional Whisper-Large speech embeddings.
- **`crates/longcat-core/src/latent3d.rs` & `avatar.rs`**:
  - `VideoLatent3D`: 4D $[C, T, H, W]$ latent tensor with temporal cross-fade continuation (`blend_continuation`) to eliminate color drift across multi-minute generations.
  - `AvatarAudioProcessor`: Multi-stream 16 kHz audio mixing into 1280-D `Whisper-Large` frame embeddings (`process_multi_audio`).

---

## CLI Usage

```bash
# 1. Text-to-Video (T2V) with 30-step Shifted Flow Matching (shift = 3.0)
cargo run --release -p longcat-cli -- t2v \
  --prompt "A cinematic drone shot over a neon cyberpunk harbor in the rain" \
  --output harbor_t2v.mp4 \
  --width 1280 --height 720 --frames 93 --steps 30 --shift 3.0

# 2. Drift-Free Video Continuation
cargo run --release -p longcat-cli -- continue \
  --video harbor_t2v.mp4 \
  --output harbor_extended.mp4 \
  --steps 30

# 3. LongCat-Avatar 1.5 Multi-Stream Dialog (8-Step Distilled Mode)
cargo run --release -p longcat-cli -- avatar \
  --audio speaker_a.wav --audio speaker_b.wav \
  --ref-image portrait.png \
  --output dialog_avatar.mp4 \
  --steps 8

# 4. Run 3D-RoPE & Shifted Flow-Matching Benchmark
cargo run --release -p longcat-cli -- bench --frames 93 --steps 8
```

---

## Performance Benchmarks vs. Upstream Python (`LongCat-Video`)

See [`BENCHMARKS.md`](BENCHMARKS.md) for full methodology and hardware telemetry (**47.9x aggregate speedup** on DiT token conditioning, 3D-RoPE, and Block-Sparse Attention).
