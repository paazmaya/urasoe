use anyhow::{Context, Result};
use colored::*;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::Duration;
/**
 * API interactions with Stable Diffusion for ControlNet Image Generator
 *
 * This module handles all communication with the Stable Diffusion API,
 * including image generation with ControlNet and model management.
 */
use std::path::Path;

// We'll use direct serde_json parsing instead of api_types structs for now
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

    /// Create a new StableDiffusionClient with a specified timeout
    ///
    /// # Arguments
    /// * `api_url` - Base URL for the Stable Diffusion API
    /// * `timeout_ms` - Timeout for API requests in milliseconds
    ///
    /// # Returns
    /// A new StableDiffusionClient instance with the specified timeout
    pub fn with_timeout(api_url: &str, timeout_ms: u64) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .build()
            .unwrap_or_else(|_| Client::new());
        
        Self {
            client,
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
            
            // Try to get error details for better handling
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("API error: {} - {}", status, error_text));
        }

        // Parse the response
        let response_text = response.text().await.context("Failed to get response text")?;
        
        // Check if the response contains error information in JSON
        if let Ok(error_json) = serde_json::from_str::<serde_json::Value>(&response_text) {
            if let Some(error) = error_json.get("error").and_then(|e| e.as_str()) {
                return Err(anyhow::anyhow!("API returned error: {}", error));
            }
        }

        // Try to parse as StableDiffusionResponse
        match serde_json::from_str::<StableDiffusionResponse>(&response_text) {
            Ok(result) => Ok(Some(result)),
            Err(e) => Err(anyhow::anyhow!("Failed to parse API response: {}", e))
        }
    }

    /// Fetch available ControlNet models from the API
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of available ControlNet model names
    pub async fn get_controlnet_models(&self) -> Result<Vec<String>> {
        let url = format!("{}controlnet/model_list", self.api_url);
        
        let response = self.client.get(&url)
            .send()
            .await
            .context("Failed to fetch ControlNet models")?;
            
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to get ControlNet models: {} {}", status, text));
        }
        
        let models_response = response.json::<serde_json::Value>().await?;
        
        // Extract model names from the response
        let model_names: Vec<String> = models_response["model_list"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|model| {
                // Extract model name from the JSON
                let model_name = model["model_name"].as_str()?;
                
                // Extract the base model name without path or extension
                let file_name = Path::new(model_name)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                
                // Remove the "control_" prefix if it exists
                if file_name.starts_with("control_") && file_name.contains("_sd15") {
                    let base_name = file_name
                        .strip_prefix("control_")
                        .unwrap_or(&file_name)
                        .to_string();
                    
                    // Remove the "_sd15" suffix if it exists
                    Some(base_name
                        .strip_suffix("_sd15")
                        .unwrap_or(&base_name)
                        .to_string())
                } else {
                    Some(file_name)
                }
            })
            .collect();
            
        Ok(model_names)
    }
    
    /// Fetch available ControlNet preprocessors (modules) from the API
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of available ControlNet preprocessor names
    pub async fn get_controlnet_modules(&self) -> Result<Vec<String>> {
        let url = format!("{}controlnet/module_list", self.api_url);
        
        let response = self.client.get(&url)
            .send()
            .await
            .context("Failed to fetch ControlNet modules")?;
            
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to get ControlNet modules: {} {}", status, text));
        }
        
        let modules_response = response.json::<serde_json::Value>().await?;
        
        // Extract module names from the response
        let modules = modules_response["module_list"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
            
        Ok(modules)
    }
    
    /// Fetch available SD model checkpoints from the API
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of available SD model checkpoint names
    pub async fn get_sd_models(&self) -> Result<Vec<String>> {
        let url = format!("{}sdapi/v1/sd-models", self.api_url);
        
        let response = self.client.get(&url)
            .send()
            .await
            .context("Failed to fetch SD models")?;
            
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to get SD models: {} {}", status, text));
        }
        
        let models = response.json::<Vec<serde_json::Value>>().await?;
        let model_names: Vec<String> = models.iter()
            .filter_map(|model| model["title"].as_str().map(String::from))
            .collect();
            
        Ok(model_names)
    }
    
    /// Fetch available sampler names from the API
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of available sampler names
    pub async fn get_samplers(&self) -> Result<Vec<String>> {
        let url = format!("{}sdapi/v1/samplers", self.api_url);
        
        let response = self.client.get(&url)
            .send()
            .await
            .context("Failed to fetch samplers")?;
            
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!("Failed to get samplers: {} {}", status, text));
        }
        
        let samplers = response.json::<Vec<serde_json::Value>>().await?;
        let sampler_names: Vec<String> = samplers.iter()
            .filter_map(|sampler| sampler["name"].as_str().map(String::from))
            .collect();
            
        Ok(sampler_names)
    }
    
    /// Validate configuration options against available API options
    ///
    /// # Arguments
    /// * `config` - Configuration to validate
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - List of any validation issues found, empty if all valid
    pub async fn validate_config_options(&self, config: &Config) -> Result<Vec<String>> {
        let mut issues = Vec::new();
        
        // Skip validation if disabled in config
        if !config.validate_options {
            println!("{}", "Option validation disabled in config.".blue());
            return Ok(issues);
        }
        
        println!("{}", "Validating configuration options against API...".blue());
        
        // Check if model checkpoint exists
        match self.get_sd_models().await {
            Ok(models) => {
                if !models.iter().any(|m| m == &config.checkpoint_model) {
                    issues.push(format!(
                        "Checkpoint model '{}' not found. Available models: {}", 
                        config.checkpoint_model, 
                        models.iter().take(5).cloned().collect::<Vec<_>>().join(", ")
                    ));
                }
            },
            Err(e) => println!("{} {}", "Could not validate checkpoint models:".yellow(), e),
        }
        
        // Check if sampler exists
        match self.get_samplers().await {
            Ok(samplers) => {
                if !samplers.iter().any(|s| s == &config.sampler_name) {
                    issues.push(format!(
                        "Sampler '{}' not found. Available samplers: {}", 
                        config.sampler_name,
                        samplers.join(", ")
                    ));
                }
            },
            Err(e) => println!("{} {}", "Could not validate samplers:".yellow(), e),
        }
        
        // Check if ControlNet model exists
        match self.get_controlnet_models().await {
            Ok(models) => {
                let model_name = format!("control_{}_sd15", config.model);
                if !models.iter().any(|m| m == &model_name) {
                    issues.push(format!(
                        "ControlNet model '{}' not found. Available ControlNet models: {}", 
                        model_name,
                        models.join(", ")
                    ));
                }
            },
            Err(e) => println!("{} {}", "Could not validate ControlNet models:".yellow(), e),
        }
        
        // Check if ControlNet module exists
        match self.get_controlnet_modules().await {
            Ok(modules) => {
                if !modules.iter().any(|m| m == &config.controlnet_module) {
                    issues.push(format!(
                        "ControlNet module '{}' not found. Available modules: {}", 
                        config.controlnet_module,
                        modules.join(", ")
                    ));
                }
            },
            Err(e) => println!("{} {}", "Could not validate ControlNet modules:".yellow(), e),
        }
        
        Ok(issues)
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
