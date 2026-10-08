use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoPE3DConfig {
    pub head_dim: usize,
    pub temporal_dim: usize,
    pub height_dim: usize,
    pub width_dim: usize,
    pub theta_base: f32,
}

impl Default for RoPE3DConfig {
    fn default() -> Self {
        // Typical DiT head_dim = 64 (partitioned: 16 temporal, 24 height, 24 width)
        Self {
            head_dim: 64,
            temporal_dim: 16,
            height_dim: 24,
            width_dim: 24,
            theta_base: 10000.0,
        }
    }
}

pub struct RoPE3D {
    pub config: RoPE3DConfig,
}

impl RoPE3D {
    pub fn new(config: RoPE3DConfig) -> Self {
        Self { config }
    }

    /// Compute 3D rotary angle frequencies for a grid coordinate (t, h, w)
    pub fn compute_frequencies(&self, t: usize, h: usize, w: usize) -> Vec<(f32, f32)> {
        let mut angles = Vec::with_capacity(self.config.head_dim / 2);

        // 1. Temporal angles
        let num_t_pairs = self.config.temporal_dim / 2;
        for i in 0..num_t_pairs {
            let freq = 1.0 / self.config.theta_base.powf((2 * i) as f32 / self.config.temporal_dim as f32);
            let angle = t as f32 * freq;
            angles.push((angle.cos(), angle.sin()));
        }

        // 2. Height angles
        let num_h_pairs = self.config.height_dim / 2;
        for i in 0..num_h_pairs {
            let freq = 1.0 / self.config.theta_base.powf((2 * i) as f32 / self.config.height_dim as f32);
            let angle = h as f32 * freq;
            angles.push((angle.cos(), angle.sin()));
        }

        // 3. Width angles
        let num_w_pairs = self.config.width_dim / 2;
        for i in 0..num_w_pairs {
            let freq = 1.0 / self.config.theta_base.powf((2 * i) as f32 / self.config.width_dim as f32);
            let angle = w as f32 * freq;
            angles.push((angle.cos(), angle.sin()));
        }

        angles
    }

    /// Apply 3D RoPE rotation to a token embedding vector [head_dim]
    pub fn apply_rope(&self, token_vec: &[f32], t: usize, h: usize, w: usize) -> Vec<f32> {
        let angles = self.compute_frequencies(t, h, w);
        let mut rotated = token_vec.to_vec();

        let num_pairs = (self.config.head_dim / 2).min(angles.len());
        for p in 0..num_pairs {
            let idx1 = p * 2;
            let idx2 = p * 2 + 1;
            if idx2 < rotated.len() {
                let x1 = rotated[idx1];
                let x2 = rotated[idx2];
                let (cos, sin) = angles[p];
                rotated[idx1] = x1 * cos - x2 * sin;
                rotated[idx2] = x1 * sin + x2 * cos;
            }
        }

        rotated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_3d() {
        let rope = RoPE3D::new(RoPE3DConfig::default());
        let vec = vec![1.0; 64];
        let rotated = rope.apply_rope(&vec, 5, 12, 18);
        assert_eq!(rotated.len(), 64);
        // Vector norm should be conserved by rotation
        let norm_orig: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_rot: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm_orig - norm_rot).abs() < 1e-4);
    }
}
