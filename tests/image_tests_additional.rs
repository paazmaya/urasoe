//! Additional image module tests for urasoe

use std::fs;
use std::io::Write;
use tempfile::tempdir;
use base64::Engine;
use urasoe::image::{ImageProcessor, image_to_base64};

/// Test image_to_base64 with a valid image
#[test]
fn test_image_to_base64_valid_image() {
    // Create a temporary directory and a simple test image
    let temp_dir = tempdir().unwrap();
    let image_path = temp_dir.path().join("test.png");
    
    // Create a minimal valid PNG file (1x1 transparent pixel)
    let png_data = [
        137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82,
        0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0, 0, 0, 31, 21, 196, 137,
        0, 0, 0, 10, 73, 68, 65, 84, 120, 156, 99, 0, 1, 0, 0, 5, 0, 1, 13,
        10, 45, 180, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130
    ];
    
    fs::write(&image_path, png_data).unwrap();
    
    // Test the function
    let result = image_to_base64(&image_path);
    assert!(result.is_ok());
    
    // Verify the base64 output matches what we expect
    let base64_output = result.unwrap();
    assert!(!base64_output.is_empty());
    assert_eq!(base64_output.len() % 4, 0); // Valid base64 should be a multiple of 4    // If we want to be extra sure, decode it back and compare with the original
    let decoded = base64::engine::general_purpose::STANDARD.decode(base64_output).unwrap();
    assert_eq!(decoded, png_data);
}

/// Test getting image list with various file formats
#[test]
fn test_get_image_list_multiple_formats() {
    // Create a temporary directory with different image formats
    let temp_dir = tempdir().unwrap();
    
    // Create sample files of different types
    let files = [
        ("image1.jpg", [1, 2, 3, 4]), 
        ("image2.png", [5, 6, 7, 8]),
        ("image3.jpeg", [9, 10, 11, 12]),
        ("image4.webp", [13, 14, 15, 16]),
        ("document.pdf", [17, 18, 19, 20]),  // Non-image format
        ("text.txt", [21, 22, 23, 24])       // Non-image format
    ];
    
    // Create the files
    for (name, content) in &files {
        let file_path = temp_dir.path().join(name);
        let mut file = fs::File::create(&file_path).unwrap();
        file.write_all(content).unwrap();
    }
    
    // Get images list
    let images = ImageProcessor::get_image_list(temp_dir.path().to_str().unwrap()).unwrap();
    
    // Should have 4 image files (jpg, png, jpeg, webp)
    assert_eq!(images.len(), 4);
    
    // Convert paths to strings for easier comparison
    let image_paths: Vec<String> = images
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
        .collect();
    
    // Verify all image files are included
    assert!(image_paths.contains(&"image1.jpg".to_string()));
    assert!(image_paths.contains(&"image2.png".to_string()));
    assert!(image_paths.contains(&"image3.jpeg".to_string()));
    assert!(image_paths.contains(&"image4.webp".to_string()));
    
    // Verify non-image files are excluded
    assert!(!image_paths.contains(&"document.pdf".to_string()));
    assert!(!image_paths.contains(&"text.txt".to_string()));
}

/// Test image_to_base64 with corrupted image
#[test]
fn test_image_to_base64_corrupted_image() {
    let temp_dir = tempdir().unwrap();
    let image_path = temp_dir.path().join("corrupted.png");
    
    // Create a file that has a PNG header but is corrupted
    let corrupted_png_data = [
        137, 80, 78, 71, 13, 10, 26, 10, // PNG signature
        0, 0, 0, 13, // Corrupted data
        99, 111, 114, 114, 117, 112, 116, 101, 100 // Random data
    ];
    
    fs::write(&image_path, corrupted_png_data).unwrap();
    
    // This should still succeed as we're just reading bytes and encoding
    let result = image_to_base64(&image_path);
    assert!(result.is_ok());
    
    // The encoded output should match the corrupted input when decoded
    let base64_output = result.unwrap();
    let decoded = base64::engine::general_purpose::STANDARD.decode(base64_output).unwrap();
    assert_eq!(decoded, corrupted_png_data);
}

/// Test directory with mixed content (images and subdirectories)
#[test]
fn test_get_image_list_with_subdirectories() {
    let temp_dir = tempdir().unwrap();
    
    // Create a subdirectory
    let sub_dir = temp_dir.path().join("subdir");
    fs::create_dir(&sub_dir).unwrap();
    
    // Create an image in the subdirectory (should not be included)
    let sub_image = sub_dir.join("sub_image.png");
    fs::File::create(&sub_image).unwrap().write_all(&[1, 2, 3, 4]).unwrap();
    
    // Create an image in the main directory (should be included)
    let main_image = temp_dir.path().join("main_image.png");
    fs::File::create(&main_image).unwrap().write_all(&[5, 6, 7, 8]).unwrap();
    
    // Get image list
    let images = ImageProcessor::get_image_list(temp_dir.path().to_str().unwrap()).unwrap();
    
    // Should only include the image in the main directory
    assert_eq!(images.len(), 1);
    assert_eq!(
        images[0].file_name().unwrap().to_str().unwrap(),
        "main_image.png"
    );
}
