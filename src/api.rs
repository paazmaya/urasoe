use anyhow::{Context, Result};
use colored::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
/**
 * API interactions with Stable Diffusion for ControlNet Image Generator
 *
 * This module handles all communication with the Stable Diffusion API,
 * including image generation with ControlNet and model management.
 */
use std::path::Path;

use crate::config::Config;
use crate::image::image_to_base64;

/// Response from the Stable Diffusion API after image generation
///
/// Contains the generated images as base64 strings, along with
/// optional parameters and information about the generation process.
#[derive(Serialize, Deserialize, Debug)]
pub struct StableDiffusionResponse {
    /// Array of base64-encoded generated images
    pub images: Vec<String>,
    /// Optional parameters used for generation
    pub parameters: Option<serde_json::Value>,
    /// Optional information about the generation process
    pub info: Option<String>,
}

/// Client for interacting with Stable Diffusion API
///
/// Handles communication with the Automatic1111 Stable Diffusion Web UI API,
/// including model loading and image generation with ControlNet.
pub struct StableDiffusionClient {
    /// HTTP client for making API requests
    client: Client,
    /// Base URL for the Stable Diffusion API
    api_url: String,
}

impl StableDiffusionClient {
    /// Create a new StableDiffusionClient instance
    ///
    /// # Arguments
    /// * `api_url` - Base URL for the Stable Diffusion API, typically "http://127.0.0.1:7860/"
    ///
    /// # Returns
    /// A new StableDiffusionClient instance
    pub fn new(api_url: &str) -> Self {
        Self {
            client: Client::new(),
            api_url: api_url.to_string(),
        }
    }

    /// Load a specific Stable Diffusion model checkpoint
    ///
    /// Sends a request to the API to load a specific model checkpoint for image generation.
    /// This should be called before attempting to generate images to ensure the desired
    /// model is active.
    ///
    /// # Arguments
    /// * `model_name` - Name of the model checkpoint to load (e.g., "realisticVisionV51_v51VAE")
    ///
    /// # Returns
    /// * `Result<()>` - Ok if successful, Error if the request fails
    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        println!("{} {}", "Loading model:".blue(), model_name);

        let url = format!("{}options", self.api_url);

        let response = self
            .client
            .post(&url)
            .json(&json!({
                "sd_model_checkpoint": model_name
            }))
            .send()
            .await
            .context("Failed to send request to load model")?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to load model: {} {}", status, text));
        }

        Ok(())
    }
    /// Generate images using ControlNet with the specified input image
    ///
    /// Sends a request to the API to generate images using ControlNet with the provided
    /// input image and configuration settings. The input image is used as a reference
    /// for the ControlNet model to guide the image generation.
    ///
    /// # Arguments
    /// * `image_path` - Path to the input image file
    /// * `config` - Configuration settings for image generation
    ///
    /// # Returns
    /// * `Result<Option<StableDiffusionResponse>>` - The API response containing generated images if successful,
    ///   None if the API responded with an error status, or an Error if the request failed
    pub async fn generate_with_controlnet(
        &self,
        image_path: &Path,
        config: &Config,
    ) -> Result<Option<StableDiffusionResponse>> {
        let image_base64 = image_to_base64(image_path)?;

        let url = format!("{}sdapi/v1/txt2img", self.api_url);

        // Use the new configuration options for ControlNet
        let controlnet_unit = json!({
            "input_image": image_base64,
            "module": config.controlnet_module,
            "model": format!("control_{}_sd15", config.model),
            "weight": config.controlnet_weight,
            "guidance_start": 0.0,
            "guidance_end": 1.0,
            "processor_res": 512,
            "threshold_a": 64,
            "threshold_b": 64,
            "control_mode": 0,
            "resize_mode": 1, // Scale to fit
            "pixel_perfect": true,
            "enabled": true
        });

        // Use sampler_name and scheduler configuration options
        let sampler_name = if config.scheduler.is_empty() {
            config.sampler_name.clone()
        } else {
            format!("{} {}", config.sampler_name, config.scheduler)
        };

        let payload = json!({
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "batch_size": config.batch_size,
            "steps": config.steps,
            "width": config.width,
            "height": config.height,
            "cfg_scale": config.cfg,
            "sampler_name": sampler_name,
            "override_settings": {
                "sd_model_checkpoint": config.checkpoint_model,
            },
            "alwayson_scripts": {
                "controlnet": {
                    "args": [controlnet_unit]
                }
            }
        });

        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .context("API request failed")?;

        if !response.status().is_success() {
            let status = response.status();
            println!("{} {}", "API responded with status:".red(), status);
            return Ok(None);
        }

        let result: StableDiffusionResponse = response
            .json()
            .await
            .context("Failed to parse API response")?;

        Ok(Some(result))
    }
}

// Legacy API functions for backward compatibility

/// Legacy function to load a specific Stable Diffusion model checkpoint
///
/// This is a standalone function for backward compatibility.
/// New code should use the StableDiffusionClient class instead.
///
/// # Arguments
/// * `model_name` - Name of the model checkpoint to load
/// * `api_url` - Base URL for the Stable Diffusion API
///
/// # Returns
/// * `Result<()>` - Ok if successful, Error if the request fails
#[allow(dead_code)]
pub async fn load_model(model_name: &str, api_url: &str) -> Result<()> {
    let client = StableDiffusionClient::new(api_url);
    client.load_model(model_name).await
}

/// Legacy function to generate images using ControlNet
///
/// This is a standalone function for backward compatibility.
/// New code should use the StableDiffusionClient class instead.
///
/// # Arguments
/// * `_client` - Unused client parameter (for compatibility)
/// * `image_path` - Path to the input image file
/// * `config` - Configuration settings for image generation
///
/// # Returns
/// * `Result<Option<StableDiffusionResponse>>` - The API response if successful
#[allow(dead_code)]
pub async fn generate_with_controlnet(
    _client: &Client, // Underscore prefix to indicate intentional non-use
    image_path: &Path,
    config: &Config,
) -> Result<Option<StableDiffusionResponse>> {
    let sd_client = StableDiffusionClient::new(&config.sd_api_url);
    sd_client.generate_with_controlnet(image_path, config).await
}
