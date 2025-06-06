//! Additional API module tests for urasoe with wiremock for HTTP mocking

use wiremock::{MockServer, Mock, ResponseTemplate};
use wiremock::matchers::{method, path};
use serde_json::json;
use urasoe::api::StableDiffusionClient;
use urasoe::config::Config;

/// Test loading a model with successful response
#[tokio::test]
async fn test_load_model_success() {
    // Start a mock server
    let mock_server = MockServer::start().await;
    
    // Print the mock server URI
    println!("Mock server started at: {}", mock_server.uri());
    
    // Create a mock response with any payload
    Mock::given(method("POST"))
        .and(path("/options"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("content-type", "application/json")
            .set_body_json(json!({"message": "Model loaded successfully"}))
        )
        .mount(&mock_server)
        .await;
    
    // Ensure the URL ends with a slash
    let uri = format!("{}/", mock_server.uri().trim_end_matches('/'));
    println!("Using URI: {}", uri);
    
    // Create client pointing to mock server
    let client = StableDiffusionClient::new(&uri);
    
    // Call load_model
    let result = client.load_model("test_model").await;
    
    // Print error if any
    if let Err(e) = &result {
        println!("Error: {:?}", e);
    }
    
    // Should be successful
    assert!(result.is_ok());
}

/// Test generate_with_controlnet with a successful response
#[tokio::test]
async fn test_generate_with_controlnet_success() {
    // Start a mock server
    let mock_server = MockServer::start().await;
    
    println!("Mock server started at: {}", mock_server.uri());
    
    // Create a mock response with a single base64 image
    let base64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=";
    let mock_response = serde_json::json!({
        "images": [base64_image],
        "parameters": {"prompt": "test prompt"},
        "info": "Generation successful"
    });
    
    // Setup the mock - accept any JSON payload for the txt2img endpoint
    Mock::given(method("POST"))
        .and(path("/sdapi/v1/txt2img"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("content-type", "application/json")
            .set_body_json(mock_response)
        )
        .mount(&mock_server)
        .await;
    
    // Ensure the URL ends with a slash
    let uri = format!("{}/", mock_server.uri().trim_end_matches('/'));
    println!("Using URI: {}", uri);
    
    // Create a client pointing to the mock server
    let client = StableDiffusionClient::new(&uri);
    
    // Create a config with mock server URL
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.sd_api_url = uri;
    
    // Create a temporary file to use as the image path
    let temp_dir = tempfile::tempdir().unwrap();
    let image_path = temp_dir.path().join("test_image.png");
    
    // Create a minimal valid PNG file
    let png_data = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13,
        10, 45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    ];
    std::fs::write(&image_path, png_data).unwrap();
    
    // Call generate_with_controlnet
    let result = client.generate_with_controlnet(&image_path, &config).await;
    
    // Print error if any
    if let Err(e) = &result {
        println!("Error: {:?}", e);
    }
    
    // Should be successful and contain our mocked base64 image
    assert!(result.is_ok());
    let response = result.unwrap().expect("Response should be Some");
    assert_eq!(response.images.len(), 1);
    assert_eq!(response.images[0], base64_image);
}

/// Test error handling when API returns error status
#[tokio::test]
async fn test_generate_with_controlnet_api_error() {
    // Start a mock server
    let mock_server = MockServer::start().await;
    
    println!("Mock server started at: {}", mock_server.uri());
    
    // Setup a mock that returns 500
    Mock::given(method("POST"))
        .and(path("/sdapi/v1/txt2img"))
        .respond_with(ResponseTemplate::new(500)
            .set_body_string("Internal Server Error")
        )
        .mount(&mock_server)
        .await;
    
    // Ensure the URL ends with a slash
    let uri = format!("{}/", mock_server.uri().trim_end_matches('/'));
    println!("Using URI: {}", uri);
    
    // Create a client pointing to the mock server
    let client = StableDiffusionClient::new(&uri);
    
    // Create a config with mock server URL
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.sd_api_url = uri;
    
    // Create a temporary file to use as the image path
    let temp_dir = tempfile::tempdir().unwrap();
    let image_path = temp_dir.path().join("test_image.png");
    
    // Create a minimal valid PNG file
    let png_data = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13,
        10, 45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    ];
    std::fs::write(&image_path, png_data).unwrap();
      // Call generate_with_controlnet
    let result = client.generate_with_controlnet(&image_path, &config).await;
    
    // Print error if any
    if let Err(e) = &result {
        println!("Error: {:?}", e);
    }
    
    // Should be Err since the API returned an error status
    assert!(result.is_err(), "Should return error for 500 status code");
    let error_str = result.unwrap_err().to_string();
    assert!(error_str.contains("500"), "Error should mention status code");
}

/// Test invalid JSON response from API
#[tokio::test]
async fn test_generate_with_controlnet_invalid_json() {
    // Start a mock server
    let mock_server = MockServer::start().await;
    
    println!("Mock server started at: {}", mock_server.uri());
    
    // Setup a mock that returns invalid JSON
    Mock::given(method("POST"))
        .and(path("/sdapi/v1/txt2img"))
        .respond_with(ResponseTemplate::new(200)
            .insert_header("content-type", "application/json")
            .set_body_string("{not valid json")
        )
        .mount(&mock_server)
        .await;
    
    // Ensure the URL ends with a slash
    let uri = format!("{}/", mock_server.uri().trim_end_matches('/'));
    println!("Using URI: {}", uri);
    
    // Create a client pointing to the mock server
    let client = StableDiffusionClient::new(&uri);
    
    // Create a config with mock server URL
    let mut config = Config::load("nonexistent_file.yml").unwrap();
    config.sd_api_url = uri;
    
    // Create a temporary file to use as the image path
    let temp_dir = tempfile::tempdir().unwrap();
    let image_path = temp_dir.path().join("test_image.png");
    
    // Create a minimal valid PNG file
    let png_data = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13,
        10, 45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    ];
    std::fs::write(&image_path, png_data).unwrap();
    
    // Call generate_with_controlnet
    let result = client.generate_with_controlnet(&image_path, &config).await;
    
    // Print error if any
    if let Err(e) = &result {
        println!("Error: {:?}", e);
    }
    
    // Should be an error since the response couldn't be parsed
    assert!(result.is_err());
}
