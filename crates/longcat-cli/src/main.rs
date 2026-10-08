use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use longcat_core::{
    GenerationTask, LongCatConfig, LongCatPipeline, WeightsManager,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "longcat",
    version,
    about = "High-performance CLI for LongCat-Video: 13.6B Unified Diffusion Transformer for Video & Audio Avatars",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate Text-to-Video (T2V)
    T2v {
        /// Text prompt describing the video scene
        #[arg(short = 'p', long = "prompt")]
        prompt: String,

        /// Output video file path (.mp4)
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Flow matching Euler inference steps (default: 30)
        #[arg(long = "steps", default_value = "30")]
        steps: usize,

        /// Classifier-Free Guidance (CFG) scale
        #[arg(long = "guidance", default_value = "5.0")]
        guidance: f32,

        /// Flow matching shift parameter
        #[arg(long = "shift", default_value = "3.0")]
        shift: f32,

        /// Video width (e.g. 720, 1280)
        #[arg(long = "width", default_value = "720")]
        width: u32,

        /// Video height (e.g. 480, 720)
        #[arg(long = "height", default_value = "480")]
        height: u32,

        /// Total video frames (e.g. 93 for ~3s)
        #[arg(long = "frames", default_value = "93")]
        frames: usize,

        /// Video framerate in fps
        #[arg(long = "fps", default_value = "30.0")]
        fps: f32,

        /// Random seed
        #[arg(long = "seed", default_value = "42")]
        seed: u64,

        /// Pretrained checkpoint directory
        #[arg(long = "checkpoint-dir")]
        checkpoint_dir: Option<PathBuf>,
    },

    /// Generate Image-to-Video (I2V)
    I2v {
        /// Input driving reference image path
        #[arg(short = 'i', long = "image")]
        image: PathBuf,

        /// Optional motion guidance prompt
        #[arg(short = 'p', long = "prompt")]
        prompt: Option<String>,

        /// Output video file path (.mp4)
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Flow matching steps
        #[arg(long = "steps", default_value = "30")]
        steps: usize,

        /// Video frames to generate
        #[arg(long = "frames", default_value = "93")]
        frames: usize,
    },

    /// Video Continuation for seamless, drift-free long video generation
    Continue {
        /// Previous video chunk to extend
        #[arg(short = 'v', long = "video")]
        video: PathBuf,

        /// Output combined extended video path
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Flow matching steps for continuation chunk
        #[arg(long = "steps", default_value = "30")]
        steps: usize,
    },

    /// Audio-driven Avatar Video Generation (LongCat-Avatar 1.5)
    Avatar {
        /// Input driving audio file(s) (.wav). Multiple streams supported for dialog
        #[arg(short = 'a', long = "audio", required = true)]
        audio: Vec<PathBuf>,

        /// Optional portrait reference image
        #[arg(short = 'i', long = "ref-image")]
        ref_image: Option<PathBuf>,

        /// Output synchronized avatar video (.mp4)
        #[arg(short = 'o', long = "output")]
        output: PathBuf,

        /// Flow matching steps (distilled 8-step mode available)
        #[arg(long = "steps", default_value = "8")]
        steps: usize,

        /// Video framerate in fps
        #[arg(long = "fps", default_value = "25.0")]
        fps: f32,
    },

    /// Inspect a SafeTensors model checkpoint or weights file
    Inspect {
        /// Path to the .safetensors checkpoint
        #[arg(short = 'c', long = "checkpoint")]
        checkpoint: PathBuf,

        /// Print individual tensor shapes
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
    },

    /// Benchmark LongCat 3D RoPE and Flow-Matching inference throughput
    Bench {
        /// Number of frames to benchmark
        #[arg(long = "frames", default_value = "45")]
        frames: usize,

        /// Number of flow matching steps
        #[arg(long = "steps", default_value = "8")]
        steps: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::T2v {
            prompt,
            output,
            steps,
            guidance,
            shift,
            width,
            height,
            frames,
            fps,
            seed,
            checkpoint_dir,
        } => {
            println!(
                "{} LongCat-Video Text-to-Video synthesis...",
                "Starting".cyan().bold()
            );
            println!("  Prompt     : \"{}\"", prompt.yellow());
            println!("  Resolution : {}x{} @ {:.1} fps ({} frames)", width, height, fps, frames);
            println!("  Flow Steps : {} (Shift: {:.1}, CFG: {:.1})", steps, shift, guidance);
            println!("  Output File: {}", output.display().to_string().green());

            let pb = ProgressBar::new(steps as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} steps ({eta})")?
                    .progress_chars("#>-"),
            );

            let config = LongCatConfig {
                task: GenerationTask::TextToVideo,
                width,
                height,
                fps,
                num_frames: frames,
                num_inference_steps: steps,
                shift,
                guidance_scale: guidance,
                audio_guidance_scale: 3.5,
                seed,
                checkpoint_dir,
            };

            let pipeline = LongCatPipeline::new(config);
            let result = pipeline.generate_text_to_video(&prompt, &output)?;

            pb.finish_with_message("Done!");

            println!("\n{}", "Generation Complete!".green().bold());
            println!("  Total Frames  : {}", result.frames_generated);
            println!("  Resolution    : {}x{}", result.resolution.0, result.resolution.1);
            println!("  Duration      : {:.2}s", result.duration_seconds);
            println!("  Saved To      : {}", result.output_path.display().to_string().green());
        }

        Commands::I2v {
            image,
            prompt,
            output,
            steps,
            frames,
        } => {
            println!(
                "{} LongCat-Video Image-to-Video synthesis...",
                "Starting".cyan().bold()
            );
            println!("  Input Image: {}", image.display().to_string().yellow());
            if let Some(p) = &prompt {
                println!("  Prompt     : \"{}\"", p);
            }
            println!("  Flow Steps : {}", steps);
            println!("  Output File: {}", output.display().to_string().green());

            let config = LongCatConfig {
                task: GenerationTask::ImageToVideo,
                num_frames: frames,
                num_inference_steps: steps,
                ..Default::default()
            };

            let pipeline = LongCatPipeline::new(config);
            let result = pipeline.generate_image_to_video(&image, prompt.as_deref(), &output)?;

            println!("\n{}", "Image Animation Complete!".green().bold());
            println!("  Total Frames  : {}", result.frames_generated);
            println!("  Saved To      : {}", result.output_path.display().to_string().green());
        }

        Commands::Continue {
            video,
            output,
            steps,
        } => {
            println!(
                "{} LongCat-Video continuation (drift-free long video)...",
                "Starting".cyan().bold()
            );
            println!("  Prefix Video: {}", video.display().to_string().yellow());
            println!("  Output File : {}", output.display().to_string().green());

            let config = LongCatConfig {
                task: GenerationTask::VideoContinuation,
                num_inference_steps: steps,
                ..Default::default()
            };

            let pipeline = LongCatPipeline::new(config);
            let result = pipeline.generate_video_continuation(&video, &output)?;

            println!("\n{}", "Video Continuation Complete!".green().bold());
            println!("  Extended To   : {} frames", result.frames_generated);
            println!("  Saved To      : {}", result.output_path.display().to_string().green());
        }

        Commands::Avatar {
            audio,
            ref_image,
            output,
            steps,
            fps,
        } => {
            println!(
                "{} LongCat-Video-Avatar 1.5 speech-driven generation...",
                "Starting".cyan().bold()
            );
            println!("  Audio Streams : {} stream(s)", audio.len().to_string().cyan());
            for (idx, p) in audio.iter().enumerate() {
                println!("    [{}] {}", idx + 1, p.display());
            }
            if let Some(img) = &ref_image {
                println!("  Portrait Ref  : {}", img.display().to_string().yellow());
            }
            println!("  Flow Steps    : {} (Distilled 8-Step Mode)", steps);
            println!("  Output File   : {}", output.display().to_string().green());

            let mut config = LongCatConfig::for_avatar(audio.len());
            config.num_inference_steps = steps;
            config.fps = fps;

            let pipeline = LongCatPipeline::new(config);
            let result = pipeline.generate_avatar(&audio, ref_image.as_ref(), output.clone())?;

            println!("\n{}", "Avatar Generation Complete!".green().bold());
            println!("  Generated     : {} frames ({:.1} fps)", result.frames_generated, fps);
            println!("  Saved To      : {}", result.output_path.display().to_string().green());
        }

        Commands::Inspect { checkpoint, verbose } => {
            println!(
                "{} SafeTensors checkpoint: {}",
                "Inspecting".cyan().bold(),
                checkpoint.display()
            );

            let summary = WeightsManager::inspect_checkpoint(&checkpoint)?;
            println!("\n{}", "Checkpoint Summary:".bold());
            println!("  Total Tensors   : {}", summary.tensor_count.to_string().cyan());
            println!("  Total Parameters: {}", format!("{:e}", summary.total_parameters as f64).yellow());

            if verbose {
                println!("\n{}", "Tensor Metadata:".bold());
                for t in &summary.tensors {
                    println!("  - {:<45} shape: {:?}, dtype: {}", t.name, t.shape, t.dtype);
                }
            }
        }

        Commands::Bench { frames, steps } => {
            println!(
                "{} LongCat benchmark: {} frames, {} flow steps...",
                "Running".cyan().bold(),
                frames,
                steps
            );

            let start = Instant::now();
            let scheduler = longcat_core::FlowMatchingEulerScheduler::new(3.0);
            let schedule = scheduler.get_schedule(steps);
            let rope = longcat_core::RoPE3D::new(longcat_core::RoPE3DConfig::default());

            for _step in &schedule {
                for f in 0..frames {
                    let _rot = rope.apply_rope(&vec![0.1; 64], f, 30, 45);
                    let _attn = longcat_core::Attention::scaled_dot_product(&vec![0.1; 64], &vec![0.1; 64], &vec![0.1; 64], 1, 1, 64);
                }
            }

            let elapsed = start.elapsed();
            let fps = frames as f64 / elapsed.as_secs_f64();

            println!("\n{}", "Benchmark Results:".green().bold());
            println!("  Time Elapsed    : {:.3}s", elapsed.as_secs_f64());
            println!("  Processing Speed: {:.2} fps", fps);
        }
    }

    Ok(())
}
