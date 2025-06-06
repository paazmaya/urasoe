//! File utilities tests for urasoe that match the actual behavior

use std::fs;
use tempfile::tempdir;
use urasoe::api::StableDiffusionResponse;
use urasoe::config::Config;
use urasoe::file_utils::FileManager;

/// Test that the metadata file contains config values, not API response values
#[test]
fn test_metadata_uses_config_values() {
    let temp_dir = tempdir().unwrap();
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    
    // Set specific config values to test
    config.output_dir = temp_dir.path().to_string_lossy().to_string();
    config.prompt = "config prompt".to_string();
    config.negative_prompt = "config negative prompt".to_string();
    config.model = "config-model".to_string();
    config.checkpoint_model = "config-checkpoint".to_string();
    config.steps = 50;
    config.cfg = 9.5;
    
    // Input image path
    let input_path = temp_dir.path().join("input.png");
    
    // Create a response with different values than config
    let response = StableDiffusionResponse {
        images: vec!["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=".to_string()],
        parameters: Some(serde_json::json!({
            "prompt": "api response prompt", 
            "negative_prompt": "api response negative",
            "cfg_scale": 7.5,
            "steps": 30,
            "sd_model_checkpoint": "api-checkpoint"
        })),
        info: Some("Generation info".to_string()),
    };
    
    // Save the generated images
    let result = FileManager::save_generated_images(&response, &input_path, &config);
    assert!(result.is_ok());
    
    // Get the metadata file path
    let base_name = input_path.file_stem().unwrap().to_string_lossy();
    let output_subdir = temp_dir.path().join(&*base_name);
    let metadata_path = output_subdir.join(format!("{}-metadata.json", base_name));
    
    // Read and parse the metadata
    let metadata_content = fs::read_to_string(metadata_path).unwrap();
    let metadata: serde_json::Value = serde_json::from_str(&metadata_content).unwrap();
    
    // Check that the metadata uses config values, not API response values
    assert_eq!(metadata["prompt"].as_str().unwrap(), "config prompt", "Metadata should use config prompt");
    assert_eq!(metadata["negative_prompt"].as_str().unwrap(), "config negative prompt", "Metadata should use config negative prompt");
    assert_eq!(metadata["controlnet_model"].as_str().unwrap(), "config-model", "Metadata should use config model");
    assert_eq!(metadata["checkpoint_model"].as_str().unwrap(), "config-checkpoint", "Metadata should use config checkpoint");
    assert_eq!(metadata["steps"].as_u64().unwrap(), 50, "Metadata should use config steps");
    assert_eq!(metadata["cfg_scale"].as_f64().unwrap(), 9.5, "Metadata should use config cfg");
    
    // Check timestamp
    assert!(metadata.get("timestamp").is_some(), "Metadata should contain timestamp");
}

/// Test that fields from API response are not used in metadata
#[test]
fn test_response_values_not_in_metadata() {
    let temp_dir = tempdir().unwrap();
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.output_dir = temp_dir.path().to_string_lossy().to_string();
    
    // Input image path
    let input_path = temp_dir.path().join("input.png");
    
    // Create a response with values that aren't in the default config
    let response = StableDiffusionResponse {
        images: vec!["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=".to_string()],
        parameters: Some(serde_json::json!({
            "custom_field": "custom value",
            "another_custom": 123,
            "sd_model_checkpoint": "api-checkpoint"
        })),
        info: Some("Generation info".to_string()),
    };
    
    // Save the generated images
    let result = FileManager::save_generated_images(&response, &input_path, &config);
    assert!(result.is_ok());
    
    // Get the metadata file path
    let base_name = input_path.file_stem().unwrap().to_string_lossy();
    let output_subdir = temp_dir.path().join(&*base_name);
    let metadata_path = output_subdir.join(format!("{}-metadata.json", base_name));
    
    // Read the metadata content
    let metadata_content = fs::read_to_string(metadata_path).unwrap();
    
    // Check that the API response values are not in the metadata
    assert!(!metadata_content.contains("custom_field"), "API response fields should not be in metadata");
    assert!(!metadata_content.contains("custom value"), "API response values should not be in metadata");
    assert!(!metadata_content.contains("api-checkpoint"), "API response checkpoint should not be in metadata");
}
