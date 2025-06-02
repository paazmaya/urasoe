use anyhow::{Context, Result};
use clap::Parser;
use colored::*;
/**
 * Generate Images with ControlNet
 *
 * This program reads images from a folder, calls Stable Diffusion Automatic1111
 * to generate images using ControlNet, and stores the results in organized subfolders.
 * It supports various ControlNet models including canny edge, depth, and pose detection.
 */
use std::fs;

// Import modules
mod api;
mod config;
mod file_utils;
mod image;
mod processing;

use config::{Args, Config};

#[tokio::main]
async fn main() -> Result<()> {
    let args: Args = Args::parse();

    println!("{}", "ControlNet Image Generator Starting...".blue());

    // Load configuration from file
    let mut config: Config = Config::load(&args.config)?;

    // Override with command line arguments
    config.apply_args(&args);
    // Print effective configuration
    if config.verbose {
        println!("{} {}", "Using ControlNet model:".blue(), config.model);
        println!(
            "{} {}",
            "Using ControlNet module:".blue(),
            config.controlnet_module
        );
        println!(
            "{} {}",
            "ControlNet weight:".blue(),
            config.controlnet_weight
        );
        println!(
            "{} {}",
            "Using checkpoint model:".blue(),
            config.checkpoint_model
        );
        println!(
            "{} {}",
            "Using sampler:".blue(),
            format!("{} {}", config.sampler_name, config.scheduler)
        );
        println!("{} {}", "Reading images from:".blue(), config.input_dir);
        println!("{} {}", "Saving output to:".blue(), config.output_dir);
        println!("{} {}", "Batch size:".blue(), config.batch_size);
        println!(
            "{} {}",
            "Image dimensions:".blue(),
            format!("{}x{}", config.width, config.height)
        );
        println!("{} {}", "Sampling steps:".blue(), config.steps);
        println!("{} {}", "CFG scale:".blue(), config.cfg);
        println!("{} {}", "Max retries:".blue(), config.max_retries);
        println!(
            "{} {}",
            "Retry delay:".blue(),
            format!("{}ms", config.retry_delay_ms)
        );
        println!(
            "{} {}",
            "Batch break:".blue(),
            format!("{}ms", config.batch_break_ms)
        );
    }

    // Ensure output directory exists
    fs::create_dir_all(&config.output_dir).context("Failed to create output directory")?;

    // Using our improved image processor
    let image_paths: Vec<std::path::PathBuf> = image::ImageProcessor::get_image_list(&config.input_dir)?;

    if image_paths.is_empty() {
        println!("{} {}", "No images found in".red(), config.input_dir);
        return Ok(());
    }

    println!(
        "{} {} {}",
        "Found".green(),
        image_paths.len(),
        "images to process".green()
    );
    // Create Stable Diffusion client and load model
    let sd_client = api::StableDiffusionClient::new(&config.sd_api_url);
    sd_client.load_model(&config.checkpoint_model).await?;

    // Set up retry manager and batch manager
    let retry_manager =
        processing::RetryManager::with_config(config.max_retries, config.retry_delay_ms);
    let batch_manager = processing::BatchManager::with_config(
        1, // Process one image at a time
        config.batch_break_ms,
    );

    // Initialize processing statistics
    let mut stats = processing::ProcessingStats::new();
    let total_images = image_paths.len();

    // Process all images with retry logic
    for (index, image_path) in image_paths.iter().enumerate() {
        println!("{} {}", "Processing:".blue(), image_path.display()); // Use retry manager to handle potential CUDA errors
        let result = retry_manager
            .process_with_retry(&sd_client, &image_path, &config)
            .await;

        match result {
            Ok(Some(generated)) => {
                if let Ok(_) =
                    file_utils::FileManager::save_generated_images(&generated, image_path, &config)
                {
                    stats.success_count += 1;
                    stats.generated_count += generated.images.len();
                } else {
                    stats
                        .failed_paths
                        .push(image_path.to_string_lossy().to_string());
                }
            }
            _ => {
                println!(
                    "{} {}",
                    "Failed to generate images for:".red(),
                    image_path.display()
                );
                stats
                    .failed_paths
                    .push(image_path.to_string_lossy().to_string());
            }
        }

        // Take a break between batches if needed
        batch_manager.manage_batch_break(index, total_images).await;
    }

    // Display final statistics
    stats.display(total_images);

    Ok(())
}
