use std::io::Write;
use std::path::Path;
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
        config: "nonexistent_file.yml".to_string(),
    };

    // Load config from a nonexistent file to get defaults
    let config = Config::load(&args.config).unwrap();

    // Verify all defaults match what we expect
    assert_eq!(config.input_dir, "./public/images");
    assert_eq!(config.output_dir, "./generated-images");
    assert_eq!(config.batch_size, 4);
    assert_eq!(config.width, 768);
    assert_eq!(config.height, 768);
    assert_eq!(config.steps, 30);
    assert_eq!(config.cfg, 7.5);
    assert_eq!(config.model, "canny");
    assert_eq!(config.controlnet_module, "canny");
    assert_eq!(config.controlnet_weight, 0.8);
    assert_eq!(config.sampler_name, "DPM++ 2M");
    assert_eq!(config.scheduler, "Karras");
    assert_eq!(config.checkpoint_model, "realisticVisionV51_v51VAE");
    assert_eq!(config.sd_api_url, "http://127.0.0.1:7860/");
    assert_eq!(config.prompt, "karate master in dojo, high detail, realistic photography");
    assert!(config.negative_prompt.contains("deformed"));
    assert_eq!(config.max_retries, 3);
    assert_eq!(config.retry_delay_ms, 10000);
    assert_eq!(config.batch_break_ms, 15000);
    assert!(!config.verbose);
}

/// Test loading config from a YAML file
#[test]
fn test_load_from_yaml() {
    // Create a temporary file with custom config
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "input_dir: ./test_images").unwrap();
    writeln!(temp_file, "output_dir: ./test_output").unwrap();
    writeln!(temp_file, "batch_size: 2").unwrap();
    writeln!(temp_file, "model: depth").unwrap();
    
    // Get the path as a string
    let config_path = temp_file.path().to_str().unwrap();
    
    // Create args with the temp config file path
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
        config: config_path.to_string(),
    };
    
    // Load the config
    let config = Config::load(&args.config).unwrap();
    
    // Verify the loaded values match what we specified
    assert_eq!(config.input_dir, "./test_images");
    assert_eq!(config.output_dir, "./test_output");
    assert_eq!(config.batch_size, 2);
    assert_eq!(config.model, "depth");
    
    // Other values should be defaults
    assert_eq!(config.width, 768);
    assert_eq!(config.height, 768);
    assert_eq!(config.controlnet_module, "canny");  // Default, not specified in YAML
}

/// Test that command line arguments override config values
#[test]
fn test_args_override_config() {
    // Create a temporary file with base config
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "input_dir: ./base_dir").unwrap();
    writeln!(temp_file, "output_dir: ./base_output").unwrap();
    writeln!(temp_file, "batch_size: 3").unwrap();
    
    // Get the path as a string
    let config_path = temp_file.path().to_str().unwrap();
    
    // Create args with overrides
    let args = Args {
        input_dir: Some("./override_dir".to_string()),
        output_dir: None,  // Don't override this
        batch_size: Some(5),
        width: Some(512), 
        height: Some(512),
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
        config: config_path.to_string(),
    };
    
    // Load and apply args
    let mut config = Config::load(&args.config).unwrap();
    config.apply_args(&args);
    
    // Verify overrides took effect
    assert_eq!(config.input_dir, "./override_dir");  // Overridden by arg
    assert_eq!(config.output_dir, "./base_output");  // From YAML file
    assert_eq!(config.batch_size, 5);               // Overridden by arg
    assert_eq!(config.width, 512);                  // Overridden by arg
    assert_eq!(config.height, 512);                 // Overridden by arg
    
    // Other values should still be defaults
    assert_eq!(config.controlnet_module, "canny");  // Default
}

/// Test handling of invalid YAML files
#[test]
fn test_invalid_yaml() {
    // Create a temporary file with invalid YAML
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "input_dir: ./test_images").unwrap();
    writeln!(temp_file, "batch_size: not_a_number").unwrap();  // Invalid, should be a number
    
    // Get the path as a string
    let config_path = temp_file.path().to_str().unwrap();
    
    // Loading should fail
    let result = Config::load(config_path);
    assert!(result.is_err());
}

/// Test that the verbose flag works correctly
#[test]
fn test_verbose_flag() {
    // Start with a config that has verbose = false (default)
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
        config: "nonexistent_file.yml".to_string(),
    };

    // Load config from a nonexistent file to get defaults
    let mut config = Config::load(&args.config).unwrap();
    
    // Initially verbose should be false
    assert!(!config.verbose);
    
    // We could set it manually (this is typically set in main.rs)
    config.verbose = true;
    assert!(config.verbose);
}

/// Test applying multiple command line arguments
#[test]
fn test_multiple_args() {
    // Start with defaults
    let args = Args {
        input_dir: Some("./custom_input".to_string()),
        output_dir: Some("./custom_output".to_string()),
        batch_size: Some(10),
        width: Some(1024),
        height: Some(1024),
        model: Some("pose".to_string()),
        controlnet_module: Some("openpose".to_string()),
        controlnet_weight: Some(0.9),
        sampler: Some("Euler a".to_string()),
        scheduler: Some("Simple".to_string()),
        steps: Some(50),
        cfg: Some(8.0),
        max_retries: Some(5),
        retry_delay: Some(20000),
        batch_break: Some(30000),
        config: "nonexistent_file.yml".to_string(),
    };

    // Load config and apply args
    let mut config = Config::load(&args.config).unwrap();
    config.apply_args(&args);
    
    // Verify all arguments were applied
    assert_eq!(config.input_dir, "./custom_input");
    assert_eq!(config.output_dir, "./custom_output");
    assert_eq!(config.batch_size, 10);
    assert_eq!(config.width, 1024);
    assert_eq!(config.height, 1024);
    assert_eq!(config.model, "pose");
    assert_eq!(config.controlnet_module, "openpose");
    assert_eq!(config.controlnet_weight, 0.9);
    assert_eq!(config.sampler_name, "Euler a");
    assert_eq!(config.scheduler, "Simple");
    assert_eq!(config.steps, 50);
    assert_eq!(config.cfg, 8.0);
    assert_eq!(config.max_retries, 5);
    assert_eq!(config.retry_delay_ms, 20000);
    assert_eq!(config.batch_break_ms, 30000);
}

/// Test handling nonexistent config path specified in args
#[test]
fn test_nonexistent_config_path() {
    // Create args with a path that definitely doesn't exist
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
        config: "definitely_does_not_exist_95123.yml".to_string(),
    };
    
    // This should still work by falling back to defaults
    let config = Config::load(&args.config).unwrap();
    
    // Should match default values
    assert_eq!(config.input_dir, "./public/images");
    assert_eq!(config.output_dir, "./generated-images");
}

/// Test partial application of args (some fields set, others not)
#[test]
fn test_partial_args_application() {
    // Create a temporary file with base config
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "input_dir: ./base_dir").unwrap();
    writeln!(temp_file, "width: 512").unwrap();
    writeln!(temp_file, "height: 512").unwrap();
    
    // Get the path as a string
    let config_path = temp_file.path().to_str().unwrap();
    
    // Create args with just one override
    let args = Args {
        input_dir: None,  // Don't override
        output_dir: None, // Don't override
        batch_size: None, // Don't override
        width: Some(1024), // Override width only
        height: None,     // Don't override
        model: None,      // Don't override
        controlnet_module: None,
        controlnet_weight: None,
        sampler: None,
        scheduler: None,
        steps: None,
        cfg: None,
        max_retries: None,
        retry_delay: None,
        batch_break: None,
        config: config_path.to_string(),
    };
    
    // Load and apply args
    let mut config = Config::load(&args.config).unwrap();
    config.apply_args(&args);
    
    // Only width should be changed from the values in the yml file
    assert_eq!(config.input_dir, "./base_dir"); // From YAML file
    assert_eq!(config.width, 1024);            // Overridden by arg
    assert_eq!(config.height, 512);            // From YAML file
}

/// Test for DEFAULT_CONFIG_PATH constant
#[test]
fn test_default_config_path_constant() {
    assert_eq!(DEFAULT_CONFIG_PATH, "urasoe.config.yml");
}

/// Test partial configuration in YAML (some fields set, others default)
#[test]
fn test_partial_yaml_config() {
    // Create a temporary file with just a few config options
    let mut temp_file = NamedTempFile::new().unwrap();
    writeln!(temp_file, "# Just a few config options").unwrap();
    writeln!(temp_file, "input_dir: ./minimal_images").unwrap();
    writeln!(temp_file, "batch_size: 2").unwrap();
    
    let config_path = temp_file.path().to_str().unwrap();
    
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
        config: config_path.to_string(),
    };
    
    let config = Config::load(&args.config).unwrap();
    
    // Specified values should match YAML
    assert_eq!(config.input_dir, "./minimal_images");
    assert_eq!(config.batch_size, 2);
    
    // Unspecified values should be defaults
    assert_eq!(config.output_dir, "./generated-images"); // Default
    assert_eq!(config.width, 768);                      // Default
    assert_eq!(config.height, 768);                     // Default
}

/// Test that values from config file match those in urasoe.config.yml
#[test]
fn test_default_values_match_config_file() {
    // This test checks that the default values in the Config struct match 
    // those in the default config file, ensuring they remain synchronized

    // Only run this test if the default config file exists
    if Path::new(DEFAULT_CONFIG_PATH).exists() {
        // Load from the default config file
        let config_from_file = Config::load(DEFAULT_CONFIG_PATH).unwrap();
        
        // Load from defaults (by using a nonexistent file)
        let default_config = Config::load("nonexistent_config.yml").unwrap();
        
        // All values should be identical
        assert_eq!(config_from_file.input_dir, default_config.input_dir);
        assert_eq!(config_from_file.output_dir, default_config.output_dir);
        assert_eq!(config_from_file.batch_size, default_config.batch_size);
        assert_eq!(config_from_file.width, default_config.width);
        assert_eq!(config_from_file.height, default_config.height);
        assert_eq!(config_from_file.steps, default_config.steps);
        assert_eq!(config_from_file.cfg, default_config.cfg);
        assert_eq!(config_from_file.model, default_config.model);
        assert_eq!(config_from_file.controlnet_module, default_config.controlnet_module);
        assert_eq!(config_from_file.controlnet_weight, default_config.controlnet_weight);
        assert_eq!(config_from_file.sampler_name, default_config.sampler_name);
        assert_eq!(config_from_file.scheduler, default_config.scheduler);
        assert_eq!(config_from_file.checkpoint_model, default_config.checkpoint_model);
        assert_eq!(config_from_file.sd_api_url, default_config.sd_api_url);
        assert_eq!(config_from_file.prompt, default_config.prompt);
        assert_eq!(config_from_file.negative_prompt, default_config.negative_prompt);
        assert_eq!(config_from_file.max_retries, default_config.max_retries);
        assert_eq!(config_from_file.retry_delay_ms, default_config.retry_delay_ms);
        assert_eq!(config_from_file.batch_break_ms, default_config.batch_break_ms);
    }
}
