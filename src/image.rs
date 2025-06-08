use anyhow::{Context, Result};
use base64::{Engine, prelude::BASE64_STANDARD};
/**
 * Image processing utilities for ControlNet Image Generator
 *
 * This module provides functionality for working with images, including:
 * - Discovering image files in directories
 * - Converting images to base64 for API transmission
 * - Supporting various image formats like JPEG, PNG, and WEBP
 */
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Image processor for handling image-related operations
pub struct ImageProcessor;

impl ImageProcessor {
    /// Get a list of image files from the specified directory
    ///
    /// Scans a directory for files with common image extensions (.jpg, .jpeg, .png, .webp)
    /// and returns their paths.
    ///
    /// # Arguments
    /// * `directory_path` - Path to the directory containing images
    ///
    /// # Returns
    /// A Result containing a vector of PathBufs to the discovered image files
    pub fn get_image_list(directory_path: &str) -> Result<Vec<PathBuf>> {
        let entries = fs::read_dir(directory_path)
            .context(format!("Error reading directory: {}", directory_path))?;

        let image_paths: Vec<PathBuf> = entries
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                let extension = path.extension()?.to_str()?.to_lowercase();

                if ["jpg", "jpeg", "png", "webp"].contains(&extension.as_str()) {
                    Some(path)
                } else {
                    None
                }
            })
            .collect();

        Ok(image_paths)
    }

    /// Convert an image file to base64 string
    /// 
    /// Reads an image file from disk and encodes it as a base64 string.
    /// This is used for transmitting images to the Stable Diffusion API.
    ///
    /// # Arguments
    /// * `image_path` - Path to the image file
    ///
    /// # Returns
    /// A Result containing the base64-encoded string of the image
    pub fn image_to_base64(image_path: &Path) -> Result<String> {
        let mut file = fs::File::open(image_path)
            .context(format!("Error opening image: {}", image_path.display()))?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .context(format!("Error reading image: {}", image_path.display()))?;

        Ok(BASE64_STANDARD.encode(&buffer))
    }
}

// Legacy functions for backward compatibility
/// Get a list of image files from the specified directory
#[allow(dead_code)]
pub fn get_image_list(directory_path: &str) -> Result<Vec<PathBuf>> {
    ImageProcessor::get_image_list(directory_path)
}

/// Convert an image file to base64 string
pub fn image_to_base64(image_path: &Path) -> Result<String> {
    ImageProcessor::image_to_base64(image_path)
}
