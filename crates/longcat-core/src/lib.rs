//! longcat-core: Core flow-matching, 3D RoPE, and block-sparse diffusion transformer engine for LongCat-Video.

pub mod attention;
pub mod avatar;
pub mod config;
pub mod flow_match;
pub mod latent3d;
pub mod pipeline;
pub mod rope_3d;
pub mod weights;

pub use attention::{Attention, BlockSparseConfig};
pub use avatar::{AvatarAudioConditioning, AvatarAudioProcessor, WHISPER_EMBED_DIM, WHISPER_SAMPLE_RATE};
pub use config::{GenerationTask, LongCatConfig};
pub use flow_match::{FlowMatchStep, FlowMatchingEulerScheduler};
pub use latent3d::VideoLatent3D;
pub use pipeline::{LongCatPipeline, PipelineResult};
pub use rope_3d::{RoPE3D, RoPE3DConfig};
pub use weights::{CheckpointSummary, TensorInfo, WeightsManager};
