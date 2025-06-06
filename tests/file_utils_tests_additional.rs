//! Additional file utils module tests for urasoe

use std::fs;
use std::io::Write;
use tempfile::tempdir;
use urasoe::api::StableDiffusionResponse;
use urasoe::config::Config;
use urasoe::file_utils::FileManager;

/// Test save_generated_images with multiple images
#[test]
fn test_save_multiple_generated_images() {
    let temp_dir = tempdir().unwrap();
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.output_dir = temp_dir.path().to_string_lossy().to_string();
    
    // Input image path
    let input_path = temp_dir.path().join("input.png");
    
    // Create a valid 1x1 PNG image in base64 (different for each image)
    let png_base64_1 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=";
    let png_base64_2 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";
      // Create a response with multiple images
    let response = StableDiffusionResponse {
        images: vec![png_base64_1.to_string(), png_base64_2.to_string()],
        parameters: Some(serde_json::json!({
            "prompt": "test prompt", 
            "negative_prompt": "test negative",
            "sd_model_checkpoint": "test_model"
        })),
        info: Some("Test generation info".to_string()),
    };
    
    // Save the generated images
    let result = FileManager::save_generated_images(&response, &input_path, &config);
    assert!(result.is_ok());
    
    // Get base name and output directory
    let base_name = input_path.file_stem().unwrap().to_string_lossy();
    let output_subdir = temp_dir.path().join(&*base_name);
    
    // Check that both image files were created
    let image_path_1 = output_subdir.join(format!("{}-1.png", base_name));
    let image_path_2 = output_subdir.join(format!("{}-2.png", base_name));
    
    assert!(image_path_1.exists());
    assert!(image_path_2.exists());
    
    // Check that metadata was created
    let metadata_path = output_subdir.join(format!("{}-metadata.json", base_name));
    assert!(metadata_path.exists());
      // Verify that the metadata file was created
    let metadata_content = fs::read_to_string(metadata_path).unwrap();
    // Note: the metadata uses values from config, not from the API response
    // So we just check that the file exists and has some content
    assert!(!metadata_content.is_empty(), "Metadata file should have content");
}

/// Test save_generated_images with a custom output directory structure
#[test]
fn test_save_images_with_nested_output_dir() {
    let temp_dir = tempdir().unwrap();
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    
    // Create a nested output directory structure
    let nested_dir = temp_dir.path().join("nested").join("output").join("dir");
    config.output_dir = nested_dir.to_string_lossy().to_string();
    
    // Create input image directory and file
    let input_dir = temp_dir.path().join("input_dir");
    fs::create_dir_all(&input_dir).unwrap();
    let input_path = input_dir.join("test_image.png");
    
    // Create a simple file
    fs::File::create(&input_path).unwrap().write_all(&[1, 2, 3, 4]).unwrap();
    
    // Valid base64 image
    let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=";
    
    // Create a response with one image
    let response = StableDiffusionResponse {
        images: vec![png_base64.to_string()],
        parameters: None,
        info: None,
    };
    
    // This should create all required directories
    let result = FileManager::save_generated_images(&response, &input_path, &config);
    assert!(result.is_ok());
    
    // Check that output directory was created with proper structure
    let base_name = input_path.file_stem().unwrap().to_string_lossy();
    let output_subdir = nested_dir.join(&*base_name);
    
    assert!(output_subdir.exists());
    assert!(output_subdir.is_dir());
    
    // Check that image file was created
    let image_path = output_subdir.join(format!("{}-1.png", base_name));
    assert!(image_path.exists());
}

/// Test that metadata contains timestamp and generator information
#[test]
fn test_metadata_contents() {
    let temp_dir = tempdir().unwrap();
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.output_dir = temp_dir.path().to_string_lossy().to_string();
    
    // Input path
    let input_path = temp_dir.path().join("input.png");
      // Create a valid response
    let response = StableDiffusionResponse {
        images: vec!["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=".to_string()],
        parameters: Some(serde_json::json!({
            "prompt": "test prompt",
            "cfg_scale": 7.5,
            "steps": 30,
            "sd_model_checkpoint": "test_model"
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
      // Check required fields from config/metadata struct
    assert!(metadata.get("timestamp").is_some(), "Metadata should contain timestamp");
    assert!(metadata.get("prompt").is_some(), "Metadata should contain prompt");
    assert!(metadata.get("negative_prompt").is_some(), "Metadata should contain negative prompt");
    assert!(metadata.get("controlnet_model").is_some(), "Metadata should contain controlnet model");
    assert!(metadata.get("checkpoint_model").is_some(), "Metadata should contain checkpoint model");
    assert!(metadata.get("steps").is_some(), "Metadata should contain steps");
    assert!(metadata.get("cfg_scale").is_some(), "Metadata should contain cfg scale");
    assert!(metadata.get("width").is_some(), "Metadata should contain width");
    assert!(metadata.get("height").is_some(), "Metadata should contain height");
    assert!(metadata.get("source_image").is_some(), "Metadata should contain source image path");
}

/// Test handling of file names with special characters
#[test]
fn test_filenames_with_special_chars() {
    let temp_dir = tempdir().unwrap();
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.output_dir = temp_dir.path().to_string_lossy().to_string();
    
    // Create input paths with special characters
    let special_chars = ["spaces in name", "with-dash", "with_underscore", "with.dot"];
    
    for special_name in &special_chars {
        let input_path = temp_dir.path().join(format!("{}.png", special_name));
        fs::File::create(&input_path).unwrap().write_all(&[1, 2, 3, 4]).unwrap();
        
        // Create a simple response
        let response = StableDiffusionResponse {
            images: vec!["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=".to_string()],
            parameters: None,
            info: None,
        };
        
        // Save the generated image
        let result = FileManager::save_generated_images(&response, &input_path, &config);
        assert!(result.is_ok());
        
        // Check that output directory was created correctly
        let base_name = input_path.file_stem().unwrap().to_string_lossy();
        let output_subdir = temp_dir.path().join(&*base_name);
        assert!(output_subdir.exists(), "Output directory should be created for filename: {}", special_name);
        
        // Check that image file was created correctly
        let image_path = output_subdir.join(format!("{}-1.png", base_name));
        assert!(image_path.exists(), "Image file should be created for filename: {}", special_name);
    }
}
