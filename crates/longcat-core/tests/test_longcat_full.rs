use longcat_core::{
    AvatarAudioProcessor, FlowMatchingEulerScheduler,
    GenerationTask, LongCatConfig, LongCatPipeline, RoPE3D, RoPE3DConfig, VideoLatent3D,
    WeightsManager, WHISPER_EMBED_DIM, WHISPER_SAMPLE_RATE,
};
use tempfile::NamedTempFile;

#[test]
fn test_flow_matching_euler_scheduler() {
    let scheduler = FlowMatchingEulerScheduler::new(1.0);
    assert_eq!(scheduler.shift, 1.0);

    let schedule = scheduler.get_schedule(20);
    assert_eq!(schedule.len(), 20);

    // Initial sigma starts at 1.0 and steps towards 0.0
    assert_eq!(schedule[0].sigma, 1.0);
    assert!(schedule[0].sigma > schedule[19].sigma);

    let x = vec![1.0f32; 16];
    let v = vec![0.5f32; 16];
    let next_x = FlowMatchingEulerScheduler::step(&x, &v, schedule[0].dt);
    assert_eq!(next_x.len(), 16);
}

#[test]
fn test_rope_3d_properties() {
    let rope = RoPE3D::new(RoPE3DConfig::default());
    assert_eq!(rope.config.head_dim, 64);

    let freqs_0 = rope.compute_frequencies(0, 0, 0);
    let freqs_1 = rope.compute_frequencies(1, 2, 3);
    assert_eq!(freqs_0.len(), 32); // 64 / 2 pairs
    assert_eq!(freqs_1.len(), 32);

    let mut q = vec![1.0f32; 64];
    rope.apply_rope(&mut q, 1, 2, 3);
    assert_eq!(q.len(), 64);
}

#[test]
fn test_avatar_whisper_audio() {
    assert_eq!(WHISPER_SAMPLE_RATE, 16000);
    assert_eq!(WHISPER_EMBED_DIM, 1280);

    let dummy_audio = vec![0.02f32; 16000 * 2]; // 2 seconds
    let cond = AvatarAudioProcessor::process_multi_audio(&[dummy_audio], 25.0);
    assert_eq!(cond.total_frames, 50);
    assert_eq!(cond.mixed_features[0].len(), WHISPER_EMBED_DIM);
}

#[test]
fn test_latent3d_operations() {
    let mut latent = VideoLatent3D::new(4, 16, 32, 32);
    assert_eq!(latent.channels, 4);
    assert_eq!(latent.frames, 16);
    assert_eq!(latent.total_elements(), 4 * 16 * 32 * 32);

    latent.data.fill(0.5);
    let slice = latent.slice_frames(0, 8);
    assert_eq!(slice.frames, 8);
    assert_eq!(slice.total_elements(), 4 * 8 * 32 * 32);
}

#[test]
fn test_generation_tasks() {
    let tasks = [
        GenerationTask::TextToVideo,
        GenerationTask::ImageToVideo,
        GenerationTask::VideoContinuation,
        GenerationTask::AvatarSingleAudio,
        GenerationTask::AvatarMultiAudio,
    ];

    for task in tasks {
        let config = LongCatConfig {
            task,
            num_frames: 49,
            num_inference_steps: 10,
            ..Default::default()
        };
        assert_eq!(config.num_frames, 49);
    }
}

#[test]
fn test_pipeline_text_to_video_mock() {
    let config = LongCatConfig {
        task: GenerationTask::TextToVideo,
        num_frames: 16,
        num_inference_steps: 2,
        ..Default::default()
    };
    let pipeline = LongCatPipeline::new(config);
    let temp_out = NamedTempFile::new().unwrap();

    let res = pipeline.generate_text_to_video("A cat playing piano", temp_out.path()).unwrap();
    assert_eq!(res.frames_generated, 16);
}

#[test]
fn test_weights_manager_invalid_path() {
    let res = WeightsManager::inspect_checkpoint("/invalid/path/longcat.safetensors");
    assert!(res.is_err());
}
