//! Additional processing module tests for urasoe

// No external imports needed for these tests
use urasoe::processing::{RetryManager, BatchManager, ProcessingStats};

/// Test RetryManager creation and configuration
#[test]
fn test_retry_manager_creation_and_config() {
    // Default retry manager
    let default_retry_manager = RetryManager::new();
    
    // Custom retry manager
    let custom_retry_manager = RetryManager::with_config(5, 2000);
    
    // We can't directly test private fields, but we can test behavior differences
    
    // Test error detection functionality
    let standard_error = anyhow::anyhow!("Standard error message");
    let cuda_error = anyhow::anyhow!("CUDA out of memory");
    
    assert!(!default_retry_manager.is_cuda_error(&standard_error));
    assert!(default_retry_manager.is_cuda_error(&cuda_error));
    
    // Also test other GPU-related error messages
    let vram_error = anyhow::anyhow!("Not enough VRAM");
    let gpu_error = anyhow::anyhow!("GPU memory exhausted");
    
    assert!(default_retry_manager.is_cuda_error(&vram_error));
    assert!(custom_retry_manager.is_cuda_error(&gpu_error));
}

/// Test retry manager behavior with different error types
#[test]
fn test_retry_manager_error_detection() {
    let retry_manager = RetryManager::new();
    
    // Test various error messages that should be detected as CUDA/GPU errors
    let gpu_error_messages = &[
        // Core CUDA errors
        "CUDA out of memory",
        "CUDA error: out of memory",
        "CUDA allocation failed",
        "Not enough memory on device",
        
        // GPU memory errors
        "GPU memory exhausted",
        "VRAM full",
        "Not enough VRAM",
        "GPU out of memory",
        
        // Hardware-specific errors
        "NVIDIA driver error",
        "Hardware error in CUDA device",
        "Device error during compute operation",
        
        // Timeouts that may indicate GPU issues
        "Operation timed out during compute",
        "CUDA kernel execution timeout",
        
        // Memory allocation failures specific to GPU
        "Memory allocation failed on device",
        "Failed to allocate GPU memory",
        "VRAM allocation error"
    ];
    
    for &msg in gpu_error_messages {
        let error = anyhow::anyhow!("{}", msg);
        assert!(
            retry_manager.is_cuda_error(&error), 
            "Error '{}' should be detected as a CUDA/GPU error", 
            msg
        );
    }
    
    // Test various error messages that should NOT be detected as CUDA/GPU errors
    let non_gpu_error_messages = &[
        // File/IO errors
        "File not found",
        "Permission denied",
        "Directory not accessible",
        
        // Network errors
        "Network timeout",
        "Connection refused",
        "Host unreachable",
        
        // Generic errors
        "Invalid argument",
        "Division by zero",
        "API returned error status",
        "Failed to parse JSON",
        
        // System memory errors (non-GPU)
        "Memory allocation error", // Generic memory error, not GPU-specific
        "Out of heap memory",      // System memory, not GPU memory
        "System memory exhausted",
        "Failed to allocate heap memory",
        
        // Other errors
        "Invalid configuration",
        "Unsupported operation",
        "Operation canceled"
    ];
    
    for &msg in non_gpu_error_messages {
        let error = anyhow::anyhow!("{}", msg);
        assert!(
            !retry_manager.is_cuda_error(&error),
            "Error '{}' should NOT be detected as a CUDA/GPU error",
            msg
        );
    }
}

/// Test batch manager creation and configuration
#[test]
fn test_batch_manager_creation_and_config() {
    // Default batch manager
    let _default_batch_manager = BatchManager::new();
    
    // Custom batch manager
    let _custom_batch_manager = BatchManager::with_config(5, 2000);
    
    // We can't directly test the batch delay since it requires async execution
    // but we can at least verify that creation doesn't panic
}

/// Test processing stats creation and methods
#[test]
fn test_processing_stats_methods() {
    let mut stats = ProcessingStats::new();
    
    // Initial state
    assert_eq!(stats.success_count, 0);
    assert_eq!(stats.generated_count, 0);
    assert_eq!(stats.failed_paths.len(), 0);
    
    // Update stats
    stats.success_count = 3;
    stats.generated_count = 12; // 3 successes with 4 images each
    stats.failed_paths.push("path/to/file1.jpg".to_string());
    stats.failed_paths.push("path/to/file2.jpg".to_string());
    
    // Test display - this just makes sure it doesn't crash
    stats.display(5);
    
    // Test adding more failures
    stats.failed_paths.push("path/to/file3.jpg".to_string());
    assert_eq!(stats.failed_paths.len(), 3);
}

/// Test processing stats with empty state
#[test]
fn test_processing_stats_empty() {
    let stats = ProcessingStats::new();
    
    // Test display with empty stats - should not panic
    stats.display(0);
}

/// Test that batch processing respects the batch size parameter
#[tokio::test]
async fn test_batch_manager_check() {
    let batch_manager = BatchManager::with_config(2, 100);
    
    // Simulate processing 5 items with batch size 2
    let mut batch_breaks = 0;
    
    for i in 0..5 {
        if batch_manager.should_take_break(i).await {
            batch_breaks += 1;
        }
    }
    
    // Should take a break after item 1 (i=1, making 2 items) and after item 3 (i=3, making 4 items)
    // Item 0 (first item) should not trigger a break
    // So we expect 2 breaks in total when processing 5 items with batch size 2
    assert_eq!(batch_breaks, 2);
}
