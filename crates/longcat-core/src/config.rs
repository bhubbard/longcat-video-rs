use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationTask {
    TextToVideo,
    ImageToVideo,
    VideoContinuation,
    AvatarSingleAudio,
    AvatarMultiAudio,
}

impl FromStr for GenerationTask {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "t2v" | "text-to-video" | "text_to_video" => Ok(GenerationTask::TextToVideo),
            "i2v" | "image-to-video" | "image_to_video" => Ok(GenerationTask::ImageToVideo),
            "continue" | "continuation" | "video-continuation" => Ok(GenerationTask::VideoContinuation),
            "avatar" | "avatar-single" | "avatar_single" => Ok(GenerationTask::AvatarSingleAudio),
            "avatar-multi" | "avatar_multi" => Ok(GenerationTask::AvatarMultiAudio),
            other => anyhow::bail!("Unknown generation task: {}", other),
        }
    }
}

impl std::fmt::Display for GenerationTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            GenerationTask::TextToVideo => "Text-to-Video (T2V)",
            GenerationTask::ImageToVideo => "Image-to-Video (I2V)",
            GenerationTask::VideoContinuation => "Video Continuation",
            GenerationTask::AvatarSingleAudio => "Avatar (Single-Audio)",
            GenerationTask::AvatarMultiAudio => "Avatar (Multi-Audio)",
        };
        write!(f, "{}", name)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongCatConfig {
    pub task: GenerationTask,
    /// Video width in pixels (e.g. 720, 1280)
    pub width: u32,
    /// Video height in pixels (e.g. 480, 720)
    pub height: u32,
    /// Video framerate (default 30.0 fps)
    pub fps: f32,
    /// Total video frames to generate (e.g. 93 frames for ~3.1s, or more for continuation)
    pub num_frames: usize,
    /// Flow matching Euler discrete steps (8 for distilled avatar, 30 for full T2V)
    pub num_inference_steps: usize,
    /// Flow match shift parameter (default 1.0 or 3.0 for high resolution)
    pub shift: f32,
    /// Text Classifier-Free Guidance (CFG) scale
    pub guidance_scale: f32,
    /// Audio guidance scale for Avatar generation
    pub audio_guidance_scale: f32,
    /// Random seed for sampling reproducibility
    pub seed: u64,
    /// Pretrained model checkpoint directory
    pub checkpoint_dir: Option<PathBuf>,
}

impl Default for LongCatConfig {
    fn default() -> Self {
        Self {
            task: GenerationTask::TextToVideo,
            width: 720,
            height: 480,
            fps: 30.0,
            num_frames: 93,
            num_inference_steps: 30,
            shift: 3.0,
            guidance_scale: 5.0,
            audio_guidance_scale: 3.5,
            seed: 42,
            checkpoint_dir: None,
        }
    }
}

impl LongCatConfig {
    pub fn for_avatar(num_audio_streams: usize) -> Self {
        let task = if num_audio_streams > 1 {
            GenerationTask::AvatarMultiAudio
        } else {
            GenerationTask::AvatarSingleAudio
        };

        Self {
            task,
            width: 512,
            height: 512,
            fps: 25.0,
            num_frames: 75,
            num_inference_steps: 8, // 8-step distilled inference for Avatar 1.5
            shift: 1.0,
            guidance_scale: 3.5,
            audio_guidance_scale: 4.0,
            seed: 42,
            checkpoint_dir: None,
        }
    }
}
