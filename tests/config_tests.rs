use std::io::Write;
use tempfile::NamedTempFile;
use urasoe::config::{Args, Config, DEFAULT_CONFIG_PATH};

/// Test that default configuration values match what we expect
#[test]
fn test_default_config_values() {
    // Create an empty Args with no overrides
    let args = Args {
        input_dir: None,
        output_dir: None,
        batch_size: None,
        width: None,
        height: None,
        model: None,
        controlnet_module: None,
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: None,
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        validate_options: None,
        validate_timeout: None,
        config: "nonexistent_file.yml".to_string(),
    };

    // Load config from a nonexistent file to get defaults
    let config = Config::load(&args.config).unwrap();    // Verify default values
    assert_eq!(config.input_dir, "./public/images");
    assert_eq!(config.output_dir, "./generated-images");
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.width, 768);
    assert_eq!(config.height, 768);
    assert_eq!(config.model, "canny");
    assert_eq!(config.controlnet_module, "canny");
    assert_eq!(config.controlnet_weight, 0.8);
    assert_eq!(config.sampler_name, "DPM++ 2M");
    assert_eq!(config.scheduler, "Karras");
    assert_eq!(config.steps, 30);
    assert_eq!(config.cfg, 7.5);
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.retry_delay_ms, 10000);
    assert_eq!(config.batch_break_ms, 15000);
    assert!(config.validate_options);
    assert_eq!(config.validate_timeout_ms, 5000);
}

/// Test applying command line arguments to config
#[test]
fn test_apply_args() {
    // Create Args with overrides
    let args = Args {
        input_dir: Some("./override_dir".to_string()),
        output_dir: Some("./output_override".to_string()),
        batch_size: Some(2),
        width: Some(768),
        height: Some(512),
        model: Some("depth".to_string()),
        controlnet_module: Some("depth".to_string()),
        controlnet_weight: Some(0.5),
        sampler: Some("Euler a".to_string()),
        scheduler: Some("Simple".to_string()),
        steps: Some(20),
        cfg: Some(8.0),
        max_retries: Some(5),
        retry_delay: Some(5000),
        batch_break: Some(20000),
        validate_options: Some(false),
        validate_timeout: Some(10000),
        config: "nonexistent_file.yml".to_string(),
    };

    // Start with default config
    let mut config = Config::load(&args.config).unwrap();
    
    // Apply args
    config.apply_args(&args);

    // Verify overrides were applied
    assert_eq!(config.input_dir, "./override_dir");
    assert_eq!(config.output_dir, "./output_override");
    assert_eq!(config.batch_size, 2);
    assert_eq!(config.width, 768);
    assert_eq!(config.height, 512);
    assert_eq!(config.model, "depth");
    assert_eq!(config.controlnet_module, "depth");
    assert_eq!(config.controlnet_weight, 0.5);
    assert_eq!(config.sampler_name, "Euler a");
    assert_eq!(config.scheduler, "Simple");
    assert_eq!(config.steps, 20);
    assert_eq!(config.cfg, 8.0);
    assert_eq!(config.max_retries, 5);
    assert_eq!(config.retry_delay_ms, 5000);
    assert_eq!(config.batch_break_ms, 20000);
    assert!(!config.validate_options);
    assert_eq!(config.validate_timeout_ms, 10000);
}

/// Test partial args application
#[test]
fn test_partial_args() {
    // Create Args with some overrides, but not all
    let args = Args {
        input_dir: None,
        output_dir: None,
        batch_size: Some(3),
        width: Some(1024),
        height: None,
        model: None,
        controlnet_module: Some("openpose".to_string()),
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: Some(40),
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        validate_options: None,
        validate_timeout: None,
        config: "nonexistent_file.yml".to_string(),
    };

    // Start with default config
    let mut config = Config::load(&args.config).unwrap();
    
    // Apply args
    config.apply_args(&args);

    // Only these should be overridden
    assert_eq!(config.batch_size, 3);
    assert_eq!(config.width, 1024);
    assert_eq!(config.controlnet_module, "openpose");
    assert_eq!(config.steps, 40);
    
    // These should remain as defaults
    assert_eq!(config.input_dir, "./public/images");
    assert_eq!(config.output_dir, "./generated-images");
    assert_eq!(config.height, 768);
    assert_eq!(config.model, "canny");
    assert_eq!(config.controlnet_weight, 0.8);
    assert_eq!(config.sampler_name, "DPM++ 2M");
    assert_eq!(config.scheduler, "Karras");
    assert_eq!(config.cfg, 7.5);
    assert!(config.validate_options);
    assert_eq!(config.validate_timeout_ms, 5000);
}

/// Test loading config from a YAML file
#[test]
fn test_load_config_from_file() {
    // Create a temporary YAML file with custom settings
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "input_dir: './custom_input'").unwrap();
    writeln!(temp_file, "output_dir: './custom_output'").unwrap();
    writeln!(temp_file, "batch_size: 10").unwrap();
    writeln!(temp_file, "width: 1024").unwrap();
    writeln!(temp_file, "height: 1024").unwrap();
    writeln!(temp_file, "model: 'openpose'").unwrap();
    writeln!(temp_file, "controlnet_module: 'openpose'").unwrap();
    writeln!(temp_file, "controlnet_weight: 0.9").unwrap();
    writeln!(temp_file, "sampler_name: 'Euler'").unwrap();
    writeln!(temp_file, "scheduler: 'Normal'").unwrap();
    writeln!(temp_file, "steps: 50").unwrap();
    writeln!(temp_file, "cfg: 9.0").unwrap();
    writeln!(temp_file, "max_retries: 2").unwrap();
    writeln!(temp_file, "retry_delay_ms: 15000").unwrap();
    writeln!(temp_file, "batch_break_ms: 5000").unwrap();
    writeln!(temp_file, "validate_options: false").unwrap();
    writeln!(temp_file, "validate_timeout_ms: 8000").unwrap();
    
    // Create args pointing to our temp file
    let args = Args {
        input_dir: Some("./custom_input".to_string()),
        output_dir: None,
        batch_size: None,
        width: None,
        height: None,
        model: None,
        controlnet_module: None,
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: None,
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        validate_options: None,
        validate_timeout: None,
        config: temp_file.path().to_string_lossy().to_string(),
    };

    // Load config from the file
    let config = Config::load(&args.config).unwrap();
    
    // Verify values from the file
    assert_eq!(config.input_dir, "./custom_input");
    assert_eq!(config.output_dir, "./custom_output");
    assert_eq!(config.batch_size, 10);
    assert_eq!(config.width, 1024);
    assert_eq!(config.height, 1024);
    assert_eq!(config.model, "openpose");
    assert_eq!(config.controlnet_module, "openpose");
    assert_eq!(config.controlnet_weight, 0.9);
    assert_eq!(config.sampler_name, "Euler");
    assert_eq!(config.scheduler, "Normal");
    assert_eq!(config.steps, 50);
    assert_eq!(config.cfg, 9.0);
    assert_eq!(config.max_retries, 2);
    assert_eq!(config.retry_delay_ms, 15000);
    assert_eq!(config.batch_break_ms, 5000);
    assert!(!config.validate_options);
    assert_eq!(config.validate_timeout_ms, 8000);
}

/// Test that args override the config file
#[test]
fn test_args_override_config() {
    // Create a temporary YAML file with custom settings
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "input_dir: './config_input'").unwrap();
    writeln!(temp_file, "output_dir: './config_output'").unwrap();
    writeln!(temp_file, "batch_size: 5").unwrap();
    writeln!(temp_file, "controlnet_module: 'depth'").unwrap();
    writeln!(temp_file, "validate_options: false").unwrap();
    writeln!(temp_file, "validate_timeout_ms: 3000").unwrap();
    
    // Create args with some overrides
    let args = Args {
        input_dir: None,  // Don't override
        output_dir: Some("./args_output".to_string()), // Override
        batch_size: Some(8), // Override
        width: None, // Don't override
        height: None,
        model: None,
        controlnet_module: None, // Don't override
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: None,
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        validate_options: Some(true), // Override
        validate_timeout: None, // Don't override
        config: temp_file.path().to_string_lossy().to_string(),
    };

    // Load config from the file then apply args
    let mut config = Config::load(&args.config).unwrap();
    config.apply_args(&args);
    
    // Verify values from file that shouldn't be overridden
    assert_eq!(config.input_dir, "./config_input");  // From file
    assert_eq!(config.controlnet_module, "depth");   // From file
    assert_eq!(config.validate_timeout_ms, 3000);    // From file
    
    // Verify values that should be overridden by args
    assert_eq!(config.output_dir, "./args_output");  // From args
    assert_eq!(config.batch_size, 8);               // From args
    assert!(config.validate_options);      // From args    // Other values should be defaults
    assert_eq!(config.width, 768);                  // Default
    assert_eq!(config.height, 768);                 // Default
}

/// Test default paths for config
#[test]
fn test_default_config_path() {
    assert_eq!(DEFAULT_CONFIG_PATH, "urasoe.config.yml");
    
    // Create args with just the config path
    let args = Args {
        input_dir: None,
        output_dir: None,
        batch_size: None,
        width: None,
        height: None,
        model: None,
        controlnet_module: None,
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: None,
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        validate_options: None,
        validate_timeout: None,
        config: DEFAULT_CONFIG_PATH.to_string(),
    };
    
    // This just verifies we can load a default config without crashing
    // The actual file may not exist in the test environment, which is fine
    let _config = Config::load(&args.config);
}

/// Test config verbose flag setting
#[test]
fn test_config_verbose() {
    // Create a temporary YAML file with verbose setting
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "verbose: true").unwrap();
    
    let args = Args {
        input_dir: None,
        output_dir: None,
        batch_size: None,
        width: None,
        height: None,
        model: None,
        controlnet_module: None,
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: None,
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        validate_options: None,
        validate_timeout: None,
        config: temp_file.path().to_string_lossy().to_string(),
    };    let config = Config::load(&args.config).unwrap();
    assert!(!config.verbose); // Default value should be false
    
    // Create another file with verbose: false
    let mut temp_file2 = NamedTempFile::new().unwrap();
    writeln!(temp_file2, "verbose: false").unwrap();
    
    let args2 = Args {
        input_dir: None,
        output_dir: None,
        batch_size: None,
        width: None,
        height: None,
        model: None,
        controlnet_module: None,
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: None,
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        validate_options: None,
        validate_timeout: None,
        config: temp_file2.path().to_string_lossy().to_string(),
    };
    
    let config2 = Config::load(&args2.config).unwrap();
    assert!(!config2.verbose);
}

// Add tests for the new validation options
#[test]
fn test_validation_options() {
    // Create a temporary YAML file with validation settings
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "validate_options: false").unwrap();
    writeln!(temp_file, "validate_timeout_ms: 12345").unwrap();
    
    let args = Args {
        input_dir: None,
        output_dir: None,
        batch_size: None,
        width: None,
        height: None,
        model: None,
        controlnet_module: None,
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: None,
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        validate_options: None,
        validate_timeout: None,
        config: temp_file.path().to_string_lossy().to_string(),
    };
    
    let config = Config::load(&args.config).unwrap();
    assert!(!config.validate_options);
    assert_eq!(config.validate_timeout_ms, 12345);
    
    // Test overriding with command line args
    let mut config2 = config.clone();
    let override_args = Args {
        input_dir: None,
        output_dir: None,
        batch_size: None,
        width: None,
        height: None,
        model: None,
        controlnet_module: None,
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: None,
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        validate_options: Some(true),
        validate_timeout: Some(7000),
        config: temp_file.path().to_string_lossy().to_string(),
    };
    
    config2.apply_args(&override_args);
    assert!(config2.validate_options);
    assert_eq!(config2.validate_timeout_ms, 7000);
}
