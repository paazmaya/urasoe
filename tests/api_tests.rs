//! API module tests for urasoe

use std::path::Path;
use urasoe::api::{StableDiffusionClient, StableDiffusionResponse, load_model as legacy_load_model, generate_with_controlnet as legacy_generate_with_controlnet};
use urasoe::config::Config;
use reqwest::Client;

#[tokio::test]
async fn test_stable_diffusion_client_new() {
    let client = StableDiffusionClient::new("http://localhost:7860/");
    // Just check that the client is created (cannot access private fields)
    let _ = client;
}

#[tokio::test]
async fn test_load_model_error() {
    // Use an invalid URL to force an error
    let client = StableDiffusionClient::new("http://127.0.0.1:9999/");
    let result = client.load_model("fake_model").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_generate_with_controlnet_error() {
    let client = StableDiffusionClient::new("http://127.0.0.1:9999/");
    let config = Config::load("nonexistent_file.yml").unwrap();
    let fake_path = Path::new("not_a_real_image.png");
    let result = client.generate_with_controlnet(fake_path, &config).await;
    assert!(result.is_err() || result.as_ref().unwrap().is_none());
}

#[tokio::test]
async fn test_legacy_load_model_error() {
    let result = legacy_load_model("fake_model", "http://127.0.0.1:9999/").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_legacy_generate_with_controlnet_error() {
    let client = Client::new();
    let config = Config::load("nonexistent_file.yml").unwrap();
    let fake_path = Path::new("not_a_real_image.png");
    let result = legacy_generate_with_controlnet(&client, fake_path, &config).await;
    assert!(result.is_err() || result.as_ref().unwrap().is_none());
}
