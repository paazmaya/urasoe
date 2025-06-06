//! Tests for retry functionality with wiremock

use std::fs;
use tempfile::tempdir;
use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path};

use urasoe::api::StableDiffusionClient;
use urasoe::config::Config;
use urasoe::processing::RetryManager;

#[cfg(test)]

/// Test basic retry functionality with a simple mock
#[tokio::test]
async fn test_simple_retry() {
    // Create a temporary directory and image
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    fs::create_dir_all(&input_dir).unwrap();
    let test_image = input_dir.join("test_image.png");
    
    // Create a minimal valid PNG file
    let png_data = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13,
        10, 45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    ];
    fs::write(&test_image, png_data).unwrap();
    
    // Start a mock server
    let mock_server = MockServer::start().await;
    println!("Mock server started at: {}", mock_server.uri());
    
    // Ensure the URL ends with a slash
    let uri = format!("{}/", mock_server.uri().trim_end_matches('/'));
    println!("Using mock server URL: {}", uri);
    
    // Create config
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.sd_api_url = uri.clone();
    config.max_retries = 1; // Just one retry for this test
    config.input_dir = input_dir.to_string_lossy().to_string();
    
    // Create the client 
    let client = StableDiffusionClient::new(&uri);
    
    // Mock model loading endpoint without expectation
    Mock::given(method("POST"))
        .and(path("/options"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"message": "OK"}))
        )
        .mount(&mock_server)
        .await;
    
    // Second request will succeed
    let base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=";
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
    
    // Create retry manager
    let retry_manager = RetryManager::with_config(1, 10);
    
    // Load model first
    client.load_model("test_model").await.expect("Model load should succeed");
    
    // Process the image with retry - this should succeed on the first try
    let result = retry_manager
        .process_with_retry(&client, &test_image, &config)
        .await;
    
    // Should succeed
    assert!(result.is_ok(), "Processing should succeed");
    
    // Should have a valid response with one image
    let response = result.unwrap();
    assert!(response.is_some(), "Should have a response");
    assert_eq!(response.unwrap().images.len(), 1, "Should have one image");
}

/// Test handling when API returns an invalid JSON error
#[tokio::test]
async fn test_all_retries_fail() {
    // Create a temporary directory and image
    let temp_dir = tempdir().unwrap();
    let input_dir = temp_dir.path().join("input");
    fs::create_dir_all(&input_dir).unwrap();
    let test_image = input_dir.join("test_image.png");
    
    // Create a minimal valid PNG file
    let png_data = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13,
        10, 45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    ];
    fs::write(&test_image, png_data).unwrap();
    
    // Start a mock server
    let mock_server = MockServer::start().await;
    println!("Mock server started at: {}", mock_server.uri());
    
    // Ensure the URL ends with a slash
    let uri = format!("{}/", mock_server.uri().trim_end_matches('/'));
    println!("Using mock server URL: {}", uri);
    
    // Create config
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.sd_api_url = uri.clone();
    config.max_retries = 1; // One retry for this test
    config.input_dir = input_dir.to_string_lossy().to_string();
    
    // Create the client 
    let client = StableDiffusionClient::new(&uri);
    
    // Mock model loading endpoint
    Mock::given(method("POST"))
        .and(path("/options"))
        .respond_with(ResponseTemplate::new(200)
            .set_body_json(serde_json::json!({"message": "OK"}))
        )
        .mount(&mock_server)
        .await;
    
    // Request will return invalid JSON to cause a parse error
    Mock::given(method("POST"))
        .and(path("/sdapi/v1/txt2img"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("content-type", "application/json")
            .set_body_string("{invalid json")
        )
        .mount(&mock_server)
        .await;
    
    // Create retry manager
    let retry_manager = RetryManager::with_config(1, 10);
    
    // Load model first
    client.load_model("test_model").await.expect("Model load should succeed");
    
    // Process the image - should fail with an error since the JSON is invalid
    let result = retry_manager
        .process_with_retry(&client, &test_image, &config)
        .await;
    
    // Should be an error since the response is invalid JSON
    assert!(result.is_err(), "Should be an error when JSON is invalid");
}
