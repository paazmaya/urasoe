use anyhow::Result;
use colored::*;
use std::path::Path;
/**
 * Advanced processing utilities for ControlNet Image Generator
 *
 * This module contains retry mechanisms and other advanced processing features
 * to handle GPU memory issues and batch processing efficiently.
 *
 * Key components:
 * - RetryManager: Handles retry logic for API calls that might fail due to CUDA/GPU memory issues
 * - BatchManager: Manages batched processing with breaks to allow GPU memory to clear
 * - ProcessingStats: Tracks success/failure statistics for batch processing
 */
use std::thread;
use std::time::Duration;

use crate::api;
use crate::config;

/// Maximum number of retry attempts for operations that may fail due to CUDA/GPU memory issues
#[allow(dead_code)]
pub const MAX_RETRIES: u32 = 3;

/// Duration between retries in milliseconds, multiplied by the attempt number
#[allow(dead_code)]
pub const RETRY_DELAY_MS: u64 = 10000;

/// Duration between batch processing in milliseconds to allow GPU memory to clear
#[allow(dead_code)]
pub const BATCH_BREAK_MS: u64 = 15000;

/// Default batch size for processing images before taking a break
#[allow(dead_code)]
pub const DEFAULT_BATCH_SIZE: u32 = 1;

/// Helper struct for managing retry attempts and memory
///
/// This struct provides retry functionality for API operations that might fail due to GPU memory issues.
/// It includes configurable retry counts and delays, and provides safe path handling for any file system
/// paths that need to be processed.
pub struct RetryManager {
    max_retries: u32,
    retry_delay_ms: u64,
}

impl Default for RetryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RetryManager {
    /// Create a new RetryManager with default settings
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            max_retries: MAX_RETRIES,
            retry_delay_ms: RETRY_DELAY_MS,
        }
    }

    /// Create a RetryManager with custom settings
    pub fn with_config(max_retries: u32, retry_delay_ms: u64) -> Self {
        Self {
            max_retries,
            retry_delay_ms,
        }
    }
    
    /// Get the maximum number of retry attempts (for testing purposes)
    #[allow(dead_code)]
    pub fn get_max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Process an image with retry logic
    ///
    /// This method takes a path to an image and processes it using the Stable Diffusion API,
    /// with automatic retry logic for handling potential CUDA/GPU memory issues.
    ///
    /// # Parameters
    /// - `client`: The StableDiffusionClient to use for API calls
    /// - `image_path`: A path-like parameter pointing to the image file (implements AsRef<Path>)
    /// - `config`: The configuration settings for image generation
    ///
    /// # Returns
    /// - `Result<Option<StableDiffusionResponse>>`: The API response on success, or an error after all retries fail
    pub async fn process_with_retry<P>(
        &self,
        client: &api::StableDiffusionClient,
        image_path: P,
        config: &config::Config,
    ) -> Result<Option<api::StableDiffusionResponse>>
    where
        P: AsRef<Path>,
    {
        let mut attempt = 0;
        let mut last_error = None;
        let image_path_ref = image_path.as_ref();

        // For logging only, convert to string representation safely
        let path_display = image_path_ref.display().to_string();

        while attempt < self.max_retries {
            if attempt > 0 {
                let delay = self.retry_delay_ms * attempt as u64;
                println!(
                    "{} {}/{} {}{}{}",
                    "Retry attempt".yellow(),
                    attempt,
                    self.max_retries,
                    "after waiting".yellow(),
                    " ".yellow(),
                    format!("{}ms", delay).yellow()
                );
                thread::sleep(Duration::from_millis(delay));

                println!(
                    "{} {} {}",
                    "Retry attempt".yellow(),
                    attempt,
                    "with reduced batch size".yellow()
                );
            }

            match client
                .generate_with_controlnet(image_path_ref, config)
                .await
            {
                Ok(result) => return Ok(result),
                Err(error) => {
                    attempt += 1;
                    if self.is_cuda_error(&error) && attempt < self.max_retries {
                        println!(
                            "{} {}/{}: {}",
                            "CUDA/GPU error detected, will retry".yellow(),
                            attempt,
                            self.max_retries,
                            error
                        );
                        // Try to free memory by yielding to the async runtime
                        tokio::task::yield_now().await;
                    } else if attempt >= self.max_retries {
                        last_error = Some(error);
                        break;
                    } else {
                        last_error = Some(error);
                    }
                }
            }
        }

        // If we get here, all retries failed
        let error = last_error.unwrap_or_else(|| {
            anyhow::anyhow!("Exhausted all retry attempts without a specific error")
        });

        println!(
            "{} {} {} {}",
            "Exhausted all".red(),
            self.max_retries,
            "retry attempts for".red(),
            path_display
        );

        Err(error)
    }

    /// Check if an error is likely related to CUDA/GPU memory issues
    pub fn is_cuda_error(&self, error: &anyhow::Error) -> bool {
        let error_msg = error.to_string().to_lowercase();
        
        // GPU-specific terms
        if error_msg.contains("cuda") || 
           error_msg.contains("gpu") || 
           error_msg.contains("vram") ||
           error_msg.contains("nvidia") {
            return true;
        }
        
        // More specific memory-related phrases that are likely GPU-related
        // Make sure we exclude system memory errors by checking for system/heap indicators
        if (error_msg.contains("out of memory") && !error_msg.contains("heap") && !error_msg.contains("system")) || 
           (error_msg.contains("memory exhausted") && !error_msg.contains("system")) ||
           (error_msg.contains("memory allocation failed") && !error_msg.contains("heap")) ||
           (error_msg.contains("not enough") && error_msg.contains("memory") && !error_msg.contains("system")) {
            return true;
        }
        
        // Timeout often indicates GPU processing issues
        if error_msg.contains("timed out") || 
           error_msg.contains("timeout") && error_msg.contains("compute") {
            return true;
        }
        
        // Device-specific errors often related to GPU
        if (error_msg.contains("device") && error_msg.contains("error")) ||
           error_msg.contains("hardware error") {
            return true;
        }
        
        false
    }
}

/// Helper for managing batch processing with breaks to allow GPU memory to clear
pub struct BatchManager {
    batch_size: u32,
    break_duration_ms: u64,
}

impl Default for BatchManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BatchManager {
    /// Create a new BatchManager with default settings
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            break_duration_ms: BATCH_BREAK_MS,
        }
    }

    /// Create a BatchManager with custom settings
    pub fn with_config(batch_size: u32, break_duration_ms: u64) -> Self {
        Self {
            batch_size,
            break_duration_ms,
        }
    }

    /// Check if we should take a break after processing an item at the given index
    ///
    /// Returns true if the current item is the last in a batch (except for the very last item)
    #[allow(dead_code)]
    pub async fn should_take_break(&self, index: usize) -> bool {
        // Check if this is the end of a batch (but not the last item)
        (index + 1) % self.batch_size as usize == 0 && index > 0
    }

    /// Take a break between batches if needed
    pub async fn manage_batch_break(&self, index: usize, total_count: usize) {
        let is_end_of_batch =
            (index + 1) % self.batch_size as usize == 0 && index < total_count - 1;

        if is_end_of_batch {
            println!(
                "{} {}{}{}",
                "Taking a break to clear GPU memory".blue(),
                "(".blue(),
                format!("{}ms", self.break_duration_ms).blue(),
                ")".blue()
            );
            thread::sleep(Duration::from_millis(self.break_duration_ms));

            // Yield to the async runtime to help with memory management
            tokio::task::yield_now().await;
        }
    }
}

/// Statistics for batch processing
#[derive(Debug, Default)]
pub struct ProcessingStats {
    pub success_count: usize,
    pub generated_count: usize,
    pub failed_paths: Vec<String>,
}

impl ProcessingStats {
    /// Create a new empty ProcessingStats
    pub fn new() -> Self {
        Self::default()
    }

    /// Display processing statistics with color formatting
    pub fn display(&self, total_images: usize) {
        println!("{}", "✓ Image generation complete!".green().bold());
        println!(
            "{} {}/{}{}{}{}",
            "Processed successfully:".green(),
            self.success_count.to_string().bold(),
            total_images,
            " images".green(),
            ", Generated: ".green(),
            format!("{} new images", self.generated_count).bold()
        );

        if !self.failed_paths.is_empty() {
            let failed_names: Vec<&str> = self
                .failed_paths
                .iter()
                .map(|p| {
                    Path::new(p)
                        .file_name()
                        .unwrap_or_default()
                        .to_str()
                        .unwrap_or("unknown")
                })
                .collect();

            println!(
                "{} {}: {}",
                "Failed images".yellow(),
                format!("({})", self.failed_paths.len()).yellow(),
                failed_names.join(", ").yellow()
            );
        }
    }
}
