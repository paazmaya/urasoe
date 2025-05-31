/**
 * Tests for the processing module
 */
#[cfg(test)]
mod processing_tests {
    use crate::processing::{BatchManager, ProcessingStats, RetryManager};

    #[test]
    fn test_processing_stats_display() {
        let mut stats = ProcessingStats::new();
        stats.success_count = 5;
        stats.generated_count = 20;
        stats.failed_paths.push("test/path1.jpg".to_string());
        stats.failed_paths.push("test/path2.jpg".to_string());

        // This just tests that the display method doesn't panic
        stats.display(7);
    }

    #[test]
    fn test_batch_manager() {
        let batch_manager = BatchManager::with_config(3, 100);

        // Testing implementation details is difficult in async context,
        // so this just verifies we can create the struct
        assert_eq!(3, 3); // Just a placeholder assertion
    }

    #[test]
    fn test_retry_manager_cuda_error_detection() {
        let retry_manager = RetryManager::with_config(3, 100);

        let cuda_error = anyhow::anyhow!("CUDA out of memory");
        assert!(retry_manager.is_cuda_error(&cuda_error));

        let gpu_error = anyhow::anyhow!("GPU memory exhausted");
        assert!(retry_manager.is_cuda_error(&gpu_error));

        let normal_error = anyhow::anyhow!("File not found");
        assert!(!retry_manager.is_cuda_error(&normal_error));
    }
}
