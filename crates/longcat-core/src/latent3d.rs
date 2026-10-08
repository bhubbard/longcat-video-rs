use serde::{Deserialize, Serialize};

/// 3D Video Latent Tensor [Channels x Frames x Height x Width]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoLatent3D {
    pub channels: usize,
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl VideoLatent3D {
    pub fn new(channels: usize, frames: usize, height: usize, width: usize) -> Self {
        Self {
            channels,
            frames,
            height,
            width,
            data: vec![0.0f32; channels * frames * height * width],
        }
    }

    pub fn total_elements(&self) -> usize {
        self.channels * self.frames * self.height * self.width
    }

    /// Extract a slice of frames [start_frame..end_frame]
    pub fn slice_frames(&self, start_frame: usize, end_frame: usize) -> Self {
        let end_frame = end_frame.min(self.frames);
        let slice_len = end_frame.saturating_sub(start_frame);
        let mut sliced = Self::new(self.channels, slice_len, self.height, self.width);

        let spatial_size = self.height * self.width;
        for c in 0..self.channels {
            for f in 0..slice_len {
                let src_f = start_frame + f;
                let src_idx = (c * self.frames + src_f) * spatial_size;
                let dst_idx = (c * slice_len + f) * spatial_size;

                sliced.data[dst_idx..dst_idx + spatial_size]
                    .copy_from_slice(&self.data[src_idx..src_idx + spatial_size]);
            }
        }

        sliced
    }

    /// Blend another latent onto this latent with smooth temporal cross-fade
    /// Used in Video Continuation to eliminate color drift and seam artifacts
    pub fn blend_continuation(&mut self, next: &VideoLatent3D, overlap_frames: usize) {
        if self.channels != next.channels || self.height != next.height || self.width != next.width {
            return;
        }

        let overlap = overlap_frames.min(self.frames).min(next.frames);
        let start_blend = self.frames.saturating_sub(overlap);
        let spatial_size = self.height * self.width;

        for f in 0..overlap {
            let alpha = f as f32 / overlap as f32; // 0.0 to 1.0 smooth fade
            let curr_f = start_blend + f;

            for c in 0..self.channels {
                let curr_idx = (c * self.frames + curr_f) * spatial_size;
                let next_idx = (c * next.frames + f) * spatial_size;

                for i in 0..spatial_size {
                    self.data[curr_idx + i] =
                        (1.0 - alpha) * self.data[curr_idx + i] + alpha * next.data[next_idx + i];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latent_slice_and_blend() {
        let mut latent1 = VideoLatent3D::new(4, 10, 8, 8);
        latent1.data.fill(1.0);

        let mut latent2 = VideoLatent3D::new(4, 10, 8, 8);
        latent2.data.fill(2.0);

        latent1.blend_continuation(&latent2, 4);

        let mid_idx = (0 * 10 + 8) * 64; // near end
        assert!(latent1.data[mid_idx] > 1.0 && latent1.data[mid_idx] < 2.0);

        let sliced = latent1.slice_frames(2, 6);
        assert_eq!(sliced.frames, 4);
    }
}
