use anyhow::{Context, Result};
use base64::{Engine, prelude::BASE64_STANDARD};
use chrono::Utc;
use colored::*;
use serde::{Deserialize, Serialize};
/**
 * File operations for ControlNet Image Generator
 */
use std::fs;
use std::path::Path;

use crate::api::StableDiffusionResponse;
use crate::config::Config;

#[derive(Serialize, Deserialize, Debug)]
pub struct ImageMetadata {
    timestamp: String,
    prompt: String,
    negative_prompt: String,
    controlnet_model: String,
    checkpoint_model: String,
    steps: u32,
    cfg_scale: f32,
    width: u32,
    height: u32,
    source_image: String,
}

pub struct FileManager;

impl FileManager {
    /// Save generated images and their metadata to the output directory
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
