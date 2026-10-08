use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockSparseConfig {
    pub block_size: usize,
    pub local_blocks: usize,
    pub stride_blocks: usize,
}

impl Default for BlockSparseConfig {
    fn default() -> Self {
        Self {
            block_size: 64,
            local_blocks: 2,
            stride_blocks: 4,
        }
    }
}

pub struct Attention;

impl Attention {
    /// Compute scaled dot-product attention
    pub fn scaled_dot_product(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        seq_q: usize,
        seq_k: usize,
        dim: usize,
    ) -> Vec<f32> {
        let scale = 1.0 / (dim as f32).sqrt();
        let mut out = vec![0.0f32; seq_q * dim];

        for i in 0..seq_q {
            let q_slice = &q[i * dim..(i + 1) * dim];
            let mut scores = Vec::with_capacity(seq_k);
            let mut max_score = f32::NEG_INFINITY;

            for j in 0..seq_k {
                let k_slice = &k[j * dim..(j + 1) * dim];
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += q_slice[d] * k_slice[d];
                }
                dot *= scale;
                if dot > max_score {
                    max_score = dot;
                }
                scores.push(dot);
            }

            // Stable Softmax
            let mut sum_exp = 0.0f32;
            let mut exp_scores = Vec::with_capacity(seq_k);
            for s in scores {
                let exp_s = (s - max_score).exp();
                sum_exp += exp_s;
                exp_scores.push(exp_s);
            }

            if sum_exp > 0.0 {
                for exp_s in &mut exp_scores {
                    *exp_s /= sum_exp;
                }
            }

            let out_slice = &mut out[i * dim..(i + 1) * dim];
            for (j, weight) in exp_scores.into_iter().enumerate() {
                let v_slice = &v[j * dim..(j + 1) * dim];
                for d in 0..dim {
                    out_slice[d] += weight * v_slice[d];
                }
            }
        }

        out
    }

    /// Block-Sparse Attention for long video sequences
    pub fn block_sparse_self_attention(
        tokens: &[f32],
        num_tokens: usize,
        token_dim: usize,
        config: &BlockSparseConfig,
    ) -> Vec<f32> {
        if num_tokens == 0 || token_dim == 0 {
            return Vec::new();
        }

        let num_blocks = (num_tokens + config.block_size - 1) / config.block_size;
        let mut output = vec![0.0f32; num_tokens * token_dim];

        for b in 0..num_blocks {
            let q_start = b * config.block_size;
            let q_end = (q_start + config.block_size).min(num_tokens);
            let q_len = q_end - q_start;

            // Determine attendable key/value blocks: local window + strided global blocks
            let mut attend_blocks = Vec::new();
            for kb in 0..num_blocks {
                let diff = if b >= kb { b - kb } else { kb - b };
                if diff <= config.local_blocks || kb % config.stride_blocks == 0 {
                    attend_blocks.push(kb);
                }
            }
            attend_blocks.sort_unstable();
            attend_blocks.dedup();

            // Gather attendable tokens
            let mut kv_tokens = Vec::new();
            for &kb in &attend_blocks {
                let kb_start = kb * config.block_size;
                let kb_end = (kb_start + config.block_size).min(num_tokens);
                let slice = &tokens[kb_start * token_dim..kb_end * token_dim];
                kv_tokens.extend_from_slice(slice);
            }

            let kv_len = kv_tokens.len() / token_dim;
            let q_slice = &tokens[q_start * token_dim..q_end * token_dim];

            let block_out = Self::scaled_dot_product(
                q_slice,
                &kv_tokens,
                &kv_tokens,
                q_len,
                kv_len,
                token_dim,
            );

            output[q_start * token_dim..q_end * token_dim].copy_from_slice(&block_out);
        }

        output
    }

    /// Cross-attention between video tokens and Whisper audio embeddings (1280 dim)
    pub fn audio_cross_attention(
        video_tokens: &[f32],
        audio_embedding: &[f32],
        num_video_tokens: usize,
        token_dim: usize,
    ) -> Vec<f32> {
        let mut projected_audio = vec![0.0f32; token_dim];
        let audio_dim = audio_embedding.len();
        if audio_dim > 0 {
            for i in 0..token_dim {
                projected_audio[i] = audio_embedding[i % audio_dim];
            }
        }

        Self::scaled_dot_product(
            video_tokens,
            &projected_audio,
            &projected_audio,
            num_video_tokens,
            1,
            token_dim,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product() {
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![0.8, 0.2, 0.1, 0.9];
        let out = Attention::scaled_dot_product(&q, &k, &v, 2, 2, 2);
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_block_sparse_attention() {
        let tokens = vec![0.5f32; 128 * 16];
        let config = BlockSparseConfig {
            block_size: 32,
            local_blocks: 1,
            stride_blocks: 2,
        };
        let out = Attention::block_sparse_self_attention(&tokens, 128, 16, &config);
        assert_eq!(out.len(), 128 * 16);
    }
}
