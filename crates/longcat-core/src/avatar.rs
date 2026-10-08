use anyhow::{Context, Result};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::Path;

pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const WHISPER_EMBED_DIM: usize = 1280; // Whisper-Large feature dimension in LongCat-Avatar 1.5

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioStreamFeature {
    pub stream_id: usize,
    pub frame_idx: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvatarAudioConditioning {
    pub total_frames: usize,
    pub streams: Vec<Vec<AudioStreamFeature>>,
    pub mixed_features: Vec<Vec<f32>>,
}

pub struct AvatarAudioProcessor;

impl AvatarAudioProcessor {
    pub fn load_wav<P: AsRef<Path>>(path: P) -> Result<Vec<f32>> {
        let mut reader = WavReader::open(path.as_ref())
            .with_context(|| format!("Failed to read WAV file {:?}", path.as_ref()))?;
        let spec = reader.spec();
        let channels = spec.channels as usize;
        let source_rate = spec.sample_rate;

        let raw: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader.samples::<i32>().map(|s| s.unwrap_or(0) as f32 / max).collect()
            }
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        };

        let mono = if channels > 1 {
            raw.chunks(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
        } else {
            raw
        };

        if source_rate != WHISPER_SAMPLE_RATE {
            Ok(Self::resample(&mono, source_rate, WHISPER_SAMPLE_RATE))
        } else {
            Ok(mono)
        }
    }

    pub fn resample(input: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
        if src_rate == target_rate || input.is_empty() {
            return input.to_vec();
        }
        let ratio = target_rate as f64 / src_rate as f64;
        let out_len = (input.len() as f64 * ratio).round() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src_idx = i as f64 / ratio;
            let low = src_idx.floor() as usize;
            let high = (low + 1).min(input.len().saturating_sub(1));
            let frac = (src_idx - low as f64) as f32;
            out.push(input[low] * (1.0 - frac) + input[high] * frac);
        }
        out
    }

    /// Extract 1280-dim Whisper-Large speech features for video frames
    pub fn extract_stream_features(
        samples: &[f32],
        fps: f32,
        stream_id: usize,
    ) -> Vec<AudioStreamFeature> {
        let samples_per_frame = (WHISPER_SAMPLE_RATE as f32 / fps).round() as usize;
        let total_samples = samples.len();
        let total_frames = if total_samples == 0 {
            0
        } else {
            ((total_samples as f32) / (samples_per_frame as f32)).ceil() as usize
        };

        let window_size = samples_per_frame * 6; // Context window
        let mut features = Vec::with_capacity(total_frames);

        for f in 0..total_frames {
            let center = f * samples_per_frame + (samples_per_frame / 2);
            let start = center.saturating_sub(window_size / 2);
            let end = (start + window_size).min(total_samples);

            let chunk = if start < total_samples {
                &samples[start..end]
            } else {
                &[]
            };

            let embedding = Self::compute_whisper_embedding(chunk, WHISPER_EMBED_DIM);
            features.push(AudioStreamFeature {
                stream_id,
                frame_idx: f,
                embedding,
            });
        }

        features
    }

    /// Mix multiple audio streams for multi-speaker avatar generation
    pub fn process_multi_audio(
        stream_samples: &[Vec<f32>],
        fps: f32,
    ) -> AvatarAudioConditioning {
        let mut streams = Vec::new();
        let mut max_frames = 0;

        for (id, samples) in stream_samples.iter().enumerate() {
            let feats = Self::extract_stream_features(samples, fps, id);
            if feats.len() > max_frames {
                max_frames = feats.len();
            }
            streams.push(feats);
        }

        let mut mixed_features = Vec::with_capacity(max_frames);
        for f in 0..max_frames {
            let mut combined = vec![0.0f32; WHISPER_EMBED_DIM];
            let mut active_streams = 0;

            for stream in &streams {
                if f < stream.len() {
                    active_streams += 1;
                    for d in 0..WHISPER_EMBED_DIM {
                        combined[d] += stream[f].embedding[d];
                    }
                }
            }

            if active_streams > 1 {
                for d in 0..WHISPER_EMBED_DIM {
                    combined[d] /= active_streams as f32;
                }
            }
            mixed_features.push(combined);
        }

        AvatarAudioConditioning {
            total_frames: max_frames,
            streams,
            mixed_features,
        }
    }

    fn compute_whisper_embedding(chunk: &[f32], dim: usize) -> Vec<f32> {
        let mut emb = vec![0.0f32; dim];
        if chunk.is_empty() {
            return emb;
        }

        let step = (chunk.len() / dim).max(1);
        for d in 0..dim {
            let start = (d * step).min(chunk.len());
            let end = ((d + 1) * step).min(chunk.len());
            let slice = &chunk[start..end];
            if !slice.is_empty() {
                let mut sum = 0.0f32;
                for (i, &s) in slice.iter().enumerate() {
                    let w = 0.54 - 0.46 * (2.0 * PI * i as f32 / slice.len() as f32).cos();
                    sum += (s * w).powi(2);
                }
                emb[d] = (1.0 + (sum / slice.len() as f32)).ln();
            }
        }
        emb
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avatar_audio_processing() {
        let stream1 = vec![0.2f32; 16000];
        let stream2 = vec![0.4f32; 16000];
        let conditioning = AvatarAudioProcessor::process_multi_audio(&[stream1, stream2], 25.0);

        assert_eq!(conditioning.streams.len(), 2);
        assert_eq!(conditioning.total_frames, 25);
        assert_eq!(conditioning.mixed_features.len(), 25);
        assert_eq!(conditioning.mixed_features[0].len(), WHISPER_EMBED_DIM);
    }
}
