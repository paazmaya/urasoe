use anyhow::{Context, Result};
use base64::{Engine, prelude::BASE64_STANDARD};
use chrono::Utc;
use colored::*;
use serde::{Deserialize, Serialize};
/**
 * File operations for ControlNet Image Generator
 *
 * This module handles file system operations for the application, including:
 * - Saving generated images to the file system
 * - Creating and maintaining metadata for generated images
 * - Managing output directories and file naming conventions
 */
use std::fs;
use std::path::Path;

use crate::api::StableDiffusionResponse;
use crate::config::Config;

/// Metadata for generated images
///
/// Stores information about the generation process and parameters used,
/// which is saved alongside the generated images for reproducibility.
#[derive(Serialize, Deserialize, Debug)]
pub struct ImageMetadata {
    /// Timestamp when the image was generated
    timestamp: String,
    /// Text prompt used for image generation
    prompt: String,
    /// Negative prompt used for image generation
    negative_prompt: String,
    /// ControlNet model used (e.g., canny, depth, openpose)
    controlnet_model: String,
    /// Stable Diffusion checkpoint model used
    checkpoint_model: String,
    /// Number of diffusion steps
    steps: u32,
    /// CFG scale value used for generation
    cfg_scale: f32,
    /// Width of the generated image in pixels
    width: u32,
    /// Height of the generated image in pixels
    height: u32,
    /// Filename of the source image used for ControlNet
    source_image: String,
}

pub struct FileManager;

impl FileManager {
    /// Save generated images and their metadata to the output directory
    ///
    /// Saves the generated images from the API response to the filesystem,
    /// organizes them in directories based on the input image name, and
    /// creates a metadata JSON file with the generation parameters.
    ///
    /// # Arguments
    /// * `result` - The StableDiffusionResponse containing generated images
    /// * `input_image_path` - Path to the original input image used
    /// * `config` - Configuration settings used for generation
    ///
    /// # Returns
    /// A Result indicating success or failure of the save operation
    pub fn save_generated_images(
        result: &StableDiffusionResponse,
        input_image_path: &Path,
        config: &Config,
    ) -> Result<()> {
        if result.images.is_empty() {
            println!("{}", "No images generated to save".yellow());
            return Ok(());
        }

        let base_name = input_image_path
            .file_stem()
            .context("Failed to extract file name")?
            .to_string_lossy();

        let output_subdir = Path::new(&config.output_dir).join(&*base_name);

        // Create subdirectory for this input image if it doesn't exist
        fs::create_dir_all(&output_subdir).context("Failed to create output subdirectory")?;

        // Configuration used to create the image is stored in metadata
        let metadata = ImageMetadata {
            timestamp: Utc::now().to_rfc3339(),
            prompt: config.prompt.clone(),
            negative_prompt: config.negative_prompt.clone(),
            controlnet_model: config.model.clone(),
            checkpoint_model: config.checkpoint_model.clone(),
            steps: config.steps,
            cfg_scale: config.cfg,
            width: config.width,
            height: config.height,
            source_image: input_image_path.to_string_lossy().to_string(),
        };

        // Save metadata
        let metadata_path = output_subdir.join(format!("{}-metadata.json", base_name));
        fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)
            .context("Failed to write metadata file")?;

        // Save generated images
        for (index, image_base64) in result.images.iter().enumerate() {
            let image_data = BASE64_STANDARD
                .decode(image_base64)
                .context("Failed to decode base64 image")?;

            let output_path = output_subdir.join(format!("{}-{}.png", base_name, index + 1));
            fs::write(&output_path, image_data).context("Failed to write image file")?;

            println!("{} {}", "Saved:".green(), output_path.display());
        }

        Ok(())
    }
}

// Legacy function for backward compatibility
/// Save generated images and their metadata to the output directory
#[allow(dead_code)]
pub fn save_generated_images(
    result: &StableDiffusionResponse,
    input_image_path: &Path,
    config: &Config,
) -> Result<()> {
    FileManager::save_generated_images(result, input_image_path, config)
}
