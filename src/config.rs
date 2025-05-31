/**
 * Configuration handling for ControlNet Image Generator
 * 
 * This module provides structures and methods for reading and managing
 * configuration settings from both command line arguments and a configuration file.
 */
use std::fs;
use anyhow::{Context, Result};
use clap::Parser;
use colored::*;
use serde::{Deserialize, Serialize};

/// Default path for the configuration file
pub const DEFAULT_CONFIG_PATH: &str = "urasoe.config.yml";

/// Command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to directory containing input images
    #[arg(long)]
    pub input_dir: Option<String>,

    /// Base path for output images
    #[arg(long)]
    pub output_dir: Option<String>,

    /// Number of images to generate for each input
    #[arg(long)]
    pub batch_size: Option<u32>,

    /// Width of generated images
    #[arg(long)]
    pub width: Option<u32>,

    /// Height of generated images
    #[arg(long)]
    pub height: Option<u32>,

    /// ControlNet model to use
    #[arg(long)]
    pub model: Option<String>,

    /// ControlNet module to use (e.g., canny, depth, pose)
    #[arg(long)]
    pub controlnet_module: Option<String>,

    /// ControlNet weight (0.0-1.0)
    #[arg(long)]
    pub controlnet_weight: Option<f32>,

    /// Sampler name to use (e.g., DPM++ 2M, Euler a)
    #[arg(long)]
    pub sampler: Option<String>,

    /// Scheduler to use (e.g., Karras)
    #[arg(long)]
    pub scheduler: Option<String>,

    /// Number of sampling steps
    #[arg(long)]
    pub steps: Option<u32>,

    /// CFG scale for generation
    #[arg(long)]
    pub cfg: Option<f32>,

    /// Maximum number of retry attempts
    #[arg(long)]
    pub max_retries: Option<u32>,

    /// Delay between retries in milliseconds
    #[arg(long)]
    pub retry_delay: Option<u64>,

    /// Break duration between batches in milliseconds
    #[arg(long)]
    pub batch_break: Option<u64>,

    /// Path to config file
    #[arg(long, default_value = DEFAULT_CONFIG_PATH)]
    pub config: String,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Config {
    // Path settings
    #[serde(default = "default_input_dir")]
    /// Directory containing input images
    pub input_dir: String,
    #[serde(default = "default_output_dir")]
    /// Directory where output images will be saved
    pub output_dir: String,

    // Image generation settings
    #[serde(default = "default_batch_size")]
    /// Number of images to generate for each input
    pub batch_size: u32,
    #[serde(default = "default_width")]
    /// Width of generated images
    pub width: u32,
    #[serde(default = "default_height")]
    /// Height of generated images
    pub height: u32,
    #[serde(default = "default_steps")]
    /// Number of sampling steps
    pub steps: u32,
    #[serde(default = "default_cfg")]
    /// CFG scale for generation
    pub cfg: f32,

    // ControlNet settings
    #[serde(default = "default_model")]
    /// ControlNet model to use
    pub model: String,
    #[serde(default = "default_controlnet_module")]
    /// ControlNet module to use (e.g., canny, depth, pose)
    pub controlnet_module: String,
    #[serde(default = "default_controlnet_weight")]
    /// ControlNet weight (0.0-1.0)
    pub controlnet_weight: f32,

    // Sampler settings
    #[serde(default = "default_sampler_name")]
    /// Sampler name to use (e.g., DPM++ 2M, Euler a)
    pub sampler_name: String,
    #[serde(default = "default_sampler_index")]
    /// Scheduler to use (e.g., Karras)
    pub scheduler: String,

    // Model settings
    #[serde(default = "default_checkpoint_model")]
    /// Checkpoint model name
    pub checkpoint_model: String,

    // API settings
    #[serde(default = "default_sd_api_url")]
    /// URL for the Stable Diffusion API
    pub sd_api_url: String,

    // Prompt settings
    #[serde(default = "default_prompt")]
    /// Prompt for image generation
    pub prompt: String,
    #[serde(default = "default_negative_prompt")]
    /// Negative prompt to exclude certain features
    pub negative_prompt: String,

    // Error handling settings
    #[serde(default = "default_max_retries")]
    /// Maximum number of retry attempts
    pub max_retries: u32,
    #[serde(default = "default_retry_delay")]
    /// Delay between retries in milliseconds
    pub retry_delay_ms: u64,

    // Batch processing settings
    #[serde(default = "default_batch_break")]
    /// Break duration between batches in milliseconds
    pub batch_break_ms: u64,

    // Printing visibility
    #[serde(skip)]
    /// If true, enables verbose printing
    pub verbose: bool,
}

// Default functions for Config
pub fn default_input_dir() -> String {
    "./public/images".to_string()
}
pub fn default_output_dir() -> String {
    "./generated-images".to_string()
}
pub fn default_batch_size() -> u32 {
    4
}
pub fn default_width() -> u32 {
    768
}
pub fn default_height() -> u32 {
    768
}
pub fn default_steps() -> u32 {
    30
}
pub fn default_cfg() -> f32 {
    7.5
}
pub fn default_model() -> String {
    "canny".to_string()
}
pub fn default_controlnet_module() -> String {
    "canny".to_string()
}
pub fn default_controlnet_weight() -> f32 {
    0.8
}
pub fn default_sampler_name() -> String {
    "DPM++ 2M".to_string()
}
pub fn default_sampler_index() -> String {
    "Karras".to_string()
}
pub fn default_checkpoint_model() -> String {
    "realisticVisionV51_v51VAE".to_string()
}
pub fn default_sd_api_url() -> String {
    "http://127.0.0.1:7860/".to_string()
}
pub fn default_prompt() -> String {
    "karate master in dojo, high detail, realistic photography".to_string()
}
pub fn default_negative_prompt() -> String {
    "deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, badly drawn hands, missing limb, floating limbs, disconnected limbs, malformed hands, blurry, ((((ugly)))), (((deformed))), ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, glitchy".to_string()
}
pub fn default_max_retries() -> u32 {
    3
}
pub fn default_retry_delay() -> u64 {
    10000
}
pub fn default_batch_break() -> u64 {
    15000
}

impl Config {
    // Load config from file, with defaults if file doesn't exist
    pub fn load(config_path: &str) -> Result<Self> {
        if let Ok(file) = fs::read_to_string(config_path) {
            serde_yaml::from_str(&file).context("Failed to parse config file")
        } else {
            println!("{} {}", "Config file not found:".yellow(), config_path);
            println!("{}", "Using default configuration".yellow());
            Ok(Config {
                input_dir: default_input_dir(),
                output_dir: default_output_dir(),
                batch_size: default_batch_size(),
                width: default_width(),
                height: default_height(),
                steps: default_steps(),
                cfg: default_cfg(),
                model: default_model(),
                controlnet_module: default_controlnet_module(),
                controlnet_weight: default_controlnet_weight(),
                sampler_name: default_sampler_name(),
                scheduler: default_sampler_index(),
                checkpoint_model: default_checkpoint_model(),
                sd_api_url: default_sd_api_url(),
                prompt: default_prompt(),
                negative_prompt: default_negative_prompt(),
                max_retries: default_max_retries(),
                retry_delay_ms: default_retry_delay(),
                batch_break_ms: default_batch_break(),
                verbose: false,
            })
        }
    } // Apply command line arguments over config file values
    pub fn apply_args(&mut self, args: &Args) {
        if let Some(input_dir) = &args.input_dir {
            self.input_dir = input_dir.clone();
        }
        if let Some(output_dir) = &args.output_dir {
            self.output_dir = output_dir.clone();
        }
        if let Some(batch_size) = args.batch_size {
            self.batch_size = batch_size;
        }
        if let Some(width) = args.width {
            self.width = width;
        }
        if let Some(height) = args.height {
            self.height = height;
        }
        if let Some(model) = &args.model {
            self.model = model.clone();
        }
        if let Some(controlnet_module) = &args.controlnet_module {
            self.controlnet_module = controlnet_module.clone();
        }
        if let Some(controlnet_weight) = args.controlnet_weight {
            self.controlnet_weight = controlnet_weight;
        }
        if let Some(sampler) = &args.sampler {
            self.sampler_name = sampler.clone();
        }
        if let Some(scheduler) = &args.scheduler {
            self.scheduler = scheduler.clone();
        }
        if let Some(steps) = args.steps {
            self.steps = steps;
        }
        if let Some(cfg) = args.cfg {
            self.cfg = cfg;
        }
        if let Some(max_retries) = args.max_retries {
            self.max_retries = max_retries;
        }
        if let Some(retry_delay) = args.retry_delay {
            self.retry_delay_ms = retry_delay;
        }
        if let Some(batch_break) = args.batch_break {
            self.batch_break_ms = batch_break;
        }
    }
}
