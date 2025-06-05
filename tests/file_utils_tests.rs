//! File utils module tests for urasoe


#[test]
fn test_save_generated_images_empty() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = urasoe::config::Config::load("nonexistent_file.yml").unwrap();
    let fake_path = temp_dir.path().join("input.png");
    let result = urasoe::file_utils::FileManager::save_generated_images(&urasoe::api::StableDiffusionResponse {
        images: vec![],
        parameters: None,
        info: None,
    }, &fake_path, &config);
    assert!(result.is_ok());
}

#[test]
fn test_save_generated_images_invalid_base64() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = urasoe::config::Config::load("nonexistent_file.yml").unwrap();
    let fake_path = temp_dir.path().join("input.png");
    let result = urasoe::file_utils::FileManager::save_generated_images(&urasoe::api::StableDiffusionResponse {
        images: vec!["not_base64".to_string()],
        parameters: None,
        info: None,
    }, &fake_path, &config);
    assert!(result.is_err());
}

#[test]
fn test_save_generated_images_valid_base64() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = urasoe::config::Config::load("nonexistent_file.yml").unwrap();
    config.output_dir = temp_dir.path().to_string_lossy().to_string();
    let fake_path = temp_dir.path().join("input.png");
    // Create a valid 1x1 PNG image in base64
    let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=";
    let result = urasoe::file_utils::FileManager::save_generated_images(&urasoe::api::StableDiffusionResponse {
        images: vec![png_base64.to_string()],
        parameters: None,
        info: None,
    }, &fake_path, &config);
    assert!(result.is_ok());
    // Check that the image file was created
    let base_name = fake_path.file_stem().unwrap().to_string_lossy();
    let output_subdir = temp_dir.path().join(&*base_name);
    let image_path = output_subdir.join(format!("{}-1.png", base_name));
    assert!(image_path.exists());
}

#[test]
fn test_save_generated_images_metadata_created() {
    let temp_dir = tempfile::tempdir().unwrap();
    let mut config = urasoe::config::Config::load("nonexistent_file.yml").unwrap();
    config.output_dir = temp_dir.path().to_string_lossy().to_string();
    let fake_path = temp_dir.path().join("input.png");
    let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=";
    let result = urasoe::file_utils::FileManager::save_generated_images(&urasoe::api::StableDiffusionResponse {
        images: vec![png_base64.to_string()],
        parameters: None,
        info: None,
    }, &fake_path, &config);
    assert!(result.is_ok());
    let base_name = fake_path.file_stem().unwrap().to_string_lossy();
    let output_subdir = temp_dir.path().join(&*base_name);
    let metadata_path = output_subdir.join(format!("{}-metadata.json", base_name));
    assert!(metadata_path.exists());
    let metadata_content = std::fs::read_to_string(metadata_path).unwrap();
    assert!(metadata_content.contains("prompt"));
}

#[test]
fn test_save_generated_images_unwritable_dir() {
    use std::fs;
    let mut config = urasoe::config::Config::load("nonexistent_file.yml").unwrap();
    let fake_path = std::path::Path::new("input.png");
    let png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/w8AAgMBApUAAAAASUVORK5CYII=";

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        // Skip if running as root
        if nix::unistd::Uid::effective().is_root() {
            eprintln!("Skipping unwritable dir test: running as root");
            return;
        }
        let temp_dir = tempfile::tempdir().unwrap();
        let unwritable = temp_dir.path().join("unwritable");
        fs::create_dir(&unwritable).unwrap();
        let mut perms = fs::metadata(&unwritable).unwrap().permissions();
        perms.set_mode(0o555); // read and execute only
        fs::set_permissions(&unwritable, perms).unwrap();
        config.output_dir = unwritable.to_string_lossy().to_string();
        let result = urasoe::file_utils::FileManager::save_generated_images(&urasoe::api::StableDiffusionResponse {
            images: vec![png_base64.to_string()],
            parameters: None,
            info: None,
        }, &fake_path, &config);
        assert!(result.is_err());
    }
    #[cfg(windows)]
    {
        // On Windows, make a file readonly and use it as output_dir to simulate unwritable
        let temp_dir = tempfile::tempdir().unwrap();
        let unwritable_file = temp_dir.path().join("readonly_file");
        fs::write(&unwritable_file, b"test").unwrap();
        let mut perms = fs::metadata(&unwritable_file).unwrap().permissions();
        perms.set_readonly(true);
        fs::set_permissions(&unwritable_file, perms).unwrap();
        config.output_dir = unwritable_file.to_string_lossy().to_string();
        let result = urasoe::file_utils::FileManager::save_generated_images(&urasoe::api::StableDiffusionResponse {
            images: vec![png_base64.to_string()],
            parameters: None,
            info: None,
        }, fake_path, &config);
        assert!(result.is_err());
    }
}
