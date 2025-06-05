//! Image module tests for urasoe

use std::io::Write;
use urasoe::image::{ImageProcessor, image_to_base64};

#[test]
fn test_get_image_list_empty_dir() {
    let temp_dir = tempfile::tempdir().unwrap();
    let images = ImageProcessor::get_image_list(temp_dir.path().to_str().unwrap()).unwrap();
    assert!(images.is_empty());
}

#[test]
fn test_get_image_list_with_images() {
    let temp_dir = tempfile::tempdir().unwrap();
    let img_path = temp_dir.path().join("test.png");
    std::fs::File::create(&img_path).unwrap().write_all(&[0u8, 1, 2, 3]).unwrap();
    let images = ImageProcessor::get_image_list(temp_dir.path().to_str().unwrap()).unwrap();
    assert_eq!(images.len(), 1);
}

#[test]
fn test_image_to_base64_invalid_path() {
    let path = std::path::Path::new("not_a_real_image.png");
    let result = image_to_base64(path);
    assert!(result.is_err());
}
