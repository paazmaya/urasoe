use serde::{Deserialize, Serialize};

/// Response for API options query
#[derive(Serialize, Deserialize, Debug)]
pub struct ApiOptionsResponse {
    /// Available ControlNet models
    #[serde(default)]
    pub controlnet_models: Vec<String>,
    
    /// Available ControlNet preprocessors (modules)
    #[serde(default)]
    pub controlnet_preprocessors: Vec<String>,
    
    /// Available samplers
    #[serde(default)]
    pub samplers: Vec<String>,
    
    /// Available SD models (checkpoints)
    #[serde(default)]
    pub sd_models: Vec<String>,
    
    /// Available schedulers
    #[serde(default)]
    pub schedulers: Vec<String>,
}

/// ControlNet models info response
#[derive(Serialize, Deserialize, Debug)]
pub struct ControlNetModelsResponse {
    /// List of available model information
    pub model_list: Vec<ControlNetModelInfo>,
}

/// Information about a single ControlNet model
#[derive(Serialize, Deserialize, Debug)]
pub struct ControlNetModelInfo {
    /// Model name
    pub model_name: String,
    /// Path to the model file
    pub model_path: String,
}

/// ControlNet module (preprocessor) info response
#[derive(Serialize, Deserialize, Debug)]
pub struct ControlNetModulesResponse {
    /// List of available preprocessor modules
    pub preprocessor_list: Vec<String>,
}

/// SD Models (checkpoints) response
#[derive(Serialize, Deserialize, Debug)]
pub struct SdModelsResponse {
    /// List of available model titles
    pub title_list: Vec<String>,
}

/// Sampler information response
#[derive(Serialize, Deserialize, Debug)]
pub struct SamplersResponse {
    /// List of available samplers
    pub names: Vec<String>,
}
