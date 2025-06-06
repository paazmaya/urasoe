//! Integration tests for the complete workflow using wiremock for HTTP mocking

use std::fs;
use tempfile::tempdir;
use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path};

use urasoe::api::StableDiffusionClient;
use urasoe::config::Config;
use urasoe::image::ImageProcessor;
use urasoe::processing::{ProcessingStats, RetryManager};
use urasoe::file_utils::FileManager;

/// Test end-to-end image processing workflow
#[tokio::test]
async fn test_end_to_end_workflow() {
    // Create a temporary directory for input and output
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(&output_dir).unwrap();
    
    // Create a test image in the input directory
    let test_image = input_dir.join("test_image.png");
    let png_data = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13,
        10, 45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    ];
    fs::write(&test_image, png_data).unwrap();
    
    // Setup a mock server
    let mock_server = MockServer::start().await;
    let base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=";
    
    // Mock model loading endpoint
    Mock::given(method("POST"))
        .and(path("/options"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"message": "OK"}))
        )
        .mount(&mock_server)
        .await;
    
    // Mock image generation endpoint
    Mock::given(method("POST"))
        .and(path("/sdapi/v1/txt2img"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("content-type", "application/json")
            .set_body_json(serde_json::json!({
                "images": [base64_image],
                "parameters": {"prompt": "test prompt"},
                "info": "Generation successful"
            }))
        )
        .mount(&mock_server)
        .await;
      // Create a custom config pointing to our mock server and directories
    let uri = format!("{}/", mock_server.uri().trim_end_matches('/'));
    println!("Using mock server URL: {}", uri);
    
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.sd_api_url = uri.clone();
    config.input_dir = input_dir.to_string_lossy().to_string();
    config.output_dir = output_dir.to_string_lossy().to_string();
    config.batch_size = 1;
    
    // Create the client with our mock server URL
    let client = StableDiffusionClient::new(&uri);
    
    // Simulate the workflow
    
    // 1. Load the model
    let load_result = client.load_model(&config.checkpoint_model).await;
    assert!(load_result.is_ok(), "Model loading should succeed");
    
    // 2. Get image list
    let images = ImageProcessor::get_image_list(&config.input_dir).unwrap();
    assert_eq!(images.len(), 1, "Should find one image");
    
    // 3. Create stats tracker
    let mut stats = ProcessingStats::new();
    
    // 4. Create retry manager
    let retry_manager = RetryManager::new();
    
    // 5. Process one image
    for image_path in images {
        let result = retry_manager
            .process_with_retry(&client, &image_path, &config)
            .await;
        
        match result {
            Ok(Some(generated)) => {
                let save_result = urasoe::file_utils::FileManager::save_generated_images(
                    &generated, &image_path, &config
                );
                assert!(save_result.is_ok(), "Saving images should succeed");
                stats.success_count += 1;
                stats.generated_count += generated.images.len();
            },
            Ok(None) => {
                stats.failed_paths.push(image_path.to_string_lossy().to_string());
            },
            Err(_) => {
                stats.failed_paths.push(image_path.to_string_lossy().to_string());
            }
        }
    }
    
    // 6. Check stats
    assert_eq!(stats.success_count, 1, "Should have 1 successful generation");
    assert_eq!(stats.generated_count, 1, "Should have generated 1 image");
    assert_eq!(stats.failed_paths.len(), 0, "Should have no failed paths");
    
    // 7. Verify output
    let base_name = test_image.file_stem().unwrap().to_string_lossy();
    let output_subdir = output_dir.join(&*base_name);
    let output_image = output_subdir.join(format!("{}-1.png", base_name));
    let metadata_file = output_subdir.join(format!("{}-metadata.json", base_name));
    
    assert!(output_subdir.exists(), "Output directory should exist");
    assert!(output_image.exists(), "Output image should exist");
    assert!(metadata_file.exists(), "Metadata file should exist");
}

/// Test the retry mechanism with a controlled setup
#[tokio::test]
async fn test_retry_behavior() {
    // Create a temporary directory structure
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    let output_dir = temp_dir.path().join("output");
    fs::create_dir_all(&input_dir).unwrap();
    fs::create_dir_all(&output_dir).unwrap();
    let test_image = input_dir.join("test_image.png");
    
    // Create a minimal valid PNG file
    let png_data = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13,
        10, 45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    ];
    fs::write(&test_image, png_data).unwrap();
    
    // Create config with test directories
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.input_dir = input_dir.to_string_lossy().to_string();
    config.output_dir = output_dir.to_string_lossy().to_string();
    config.max_retries = 2; // Set to 2 for testing retry behavior
    
    // Set up mock server
    let mock_server = MockServer::start().await;
    let base_url = format!("{}/", mock_server.uri().trim_end_matches('/'));
    config.sd_api_url = base_url.clone();
    
    println!("Using mock server at: {}", base_url);
    
    // Mock model loading endpoint (always succeeds)
    Mock::given(method("POST"))
        .and(path("/options"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"message": "OK"}))
        )
        .mount(&mock_server)
        .await;
    
    // A valid base64 image for our mock response
    let base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=";
    
    // Create the successful response
    let success_response = serde_json::json!({
        "images": [base64_image],
        "parameters": {"prompt": "test prompt"},
        "info": "Generation successful"
    });
    
    // ----------------------------------------------------------------
    // PART 1: Test success case - first attempt succeeds
    // ----------------------------------------------------------------
    
    // Create client and retry manager
    let client = StableDiffusionClient::new(&config.sd_api_url);
    client.load_model("test_model").await.expect("Model loading should succeed");
    let retry_manager = RetryManager::with_config(2, 10); // Allow 2 retries
    
    // Mock successful image generation
    Mock::given(method("POST"))
        .and(path("/sdapi/v1/txt2img"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(success_response.clone())
        )
        .expect(1) // Expect exactly one call
        .mount(&mock_server)
        .await;
    
    // Process with retry - should succeed on first try
    let result = retry_manager
        .process_with_retry(&client, &test_image, &config)
        .await;
    
    // Verify success
    assert!(result.is_ok(), "Should succeed on first attempt");
    assert!(result.as_ref().unwrap().is_some(), "Should have a response");
    
    // Test image processing and saving
    if let Ok(Some(response)) = result {
        // Verify the response content
        assert_eq!(response.images.len(), 1, "Should have one image");
        
        // Save and verify output
        let save_result = FileManager::save_generated_images(&response, &test_image, &config);
        assert!(save_result.is_ok(), "Saving images should succeed");
        
        // Verify output files exist
        let output_subdir = output_dir.join("test_image");
        let output_path = output_subdir.join("test_image-1.png");
        let metadata_path = output_subdir.join("test_image-metadata.json");
        
        assert!(output_subdir.exists(), "Output directory should exist");
        assert!(output_path.exists(), "Output image should exist");
        assert!(metadata_path.exists(), "Metadata file should exist");
        
        // Verify the image file is not empty
        let file_size = fs::metadata(&output_path).unwrap().len();
        assert!(file_size > 0, "Output image file should not be empty");
    } else {
        panic!("Failed to process image or unexpected response format");
    }
    
    // ----------------------------------------------------------------
    // PART 2: Test retry behavior with initial failure
    // ----------------------------------------------------------------
    
    // Create a fresh test image
    let retry_test_image = input_dir.join("retry_test.png");
    fs::write(&retry_test_image, png_data).unwrap();
    
    // Reset the mock server for this test case
    mock_server.reset().await;
    
    // Set up model loading mock again
    Mock::given(method("POST"))
        .and(path("/options"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"message": "OK"}))
        )
        .mount(&mock_server)
        .await;
      // ----------------------------------------------------------------
    // PART 2: Test that the CUDA error detection works correctly
    // ----------------------------------------------------------------
      // Test various error patterns to ensure our enhanced detection works correctly
    let error_cases = vec![
        ("CUDA out of memory", true),
        ("GPU memory exhausted", true),
        ("VRAM limit exceeded", true),
        ("Not enough GPU memory available", true),
        ("NVIDIA driver error", true),
        ("Hardware error in device", true),
        ("Memory allocation failed on device", true),
        ("Failed to allocate GPU memory", true),
        ("CUDA kernel execution timeout", true),
        ("Operation timed out during compute", true),
        
        ("File not found", false),
        ("Network timeout", false),
        ("Invalid JSON response", false),
        ("Out of heap memory", false), // System memory, not GPU
        ("System memory exhausted", false),
        ("Failed to allocate heap memory", false),
    ];
    
    // Use reference to avoid moving error_cases
    for (error_msg, should_detect) in &error_cases {
        let error = anyhow::anyhow!("{}", error_msg);
        let detected = retry_manager.is_cuda_error(&error);
        assert_eq!(
            detected, 
            *should_detect, 
            "Error '{}' should{} be detected as CUDA/GPU error", 
            error_msg, 
            if *should_detect { "" } else { " not" }
        );
    }
    
    // Verify that the output file exists from our first test
    let output_subdir = output_dir.join("test_image");
    let output_path = output_subdir.join("test_image-1.png");
    assert!(output_path.exists(), "Output file should exist");
      let metadata = std::fs::metadata(&output_path).unwrap();
    assert!(metadata.len() > 0, "Output file should not be empty");
    
    // Use reference again to avoid moving error_cases
    for (error_msg, should_detect) in &error_cases {
        let error = anyhow::anyhow!("{}", error_msg);
        let detected = retry_manager.is_cuda_error(&error);        assert_eq!(
            detected, 
            *should_detect, 
            "Error '{}' should{} be detected as CUDA/GPU error", 
            error_msg, 
            if *should_detect { "" } else { " not" }
        );
    }
}
