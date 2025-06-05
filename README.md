# urasoe (浦添)

[![Rust CI](https://github.com/paazmaya/urasoe/actions/workflows/rust.yml/badge.svg)](https://github.com/paazmaya/urasoe/actions/workflows/rust.yml)
[![Cross-Platform Build](https://github.com/paazmaya/urasoe/actions/workflows/cross-platform-build.yml/badge.svg)](https://github.com/paazmaya/urasoe/actions/workflows/cross-platform-build.yml)
[![Security Audit](https://github.com/paazmaya/urasoe/actions/workflows/audit.yml/badge.svg)](https://github.com/paazmaya/urasoe/actions/workflows/audit.yml)
[![DeepSource](https://app.deepsource.com/gh/paazmaya/urasoe.svg/?label=active+issues&show_trend=true&token=AOeYFYo9zWVacb1YM7XOWak5)](https://app.deepsource.com/gh/paazmaya/urasoe/)

A Rust utility that reads images from a directory, then uses locally running Stable Diffusion Automatic1111 and its ControlNet plugin to generate images that mimic the shapes and features of the originals.

## Features

- Processes images from a specified directory
- Supports various ControlNet models (canny, depth, pose, etc.)
- Organizes generated images in subfolders
- Configurable image generation parameters
- Stores metadata for each generation
- Advanced GPU memory management with retry logic
- Batch processing with configurable breaks to avoid memory issues
- Comprehensive error handling and statistics

## Usage

```bash
# Build the project
cargo build --release

# Run with default options
cargo run --release

# Run with custom options
cargo run --release -- --input-dir="./my-images" --output-dir="./results" --model="depth" --batch-size=2
```

### Command Line Options

- `--input-dir` - Path to directory containing input images (default: "./public/images")
- `--output-dir` - Base path for output directories (default: "./generated-images")
- `--batch-size` - Number of images to generate for each input (default: 4)
- `--width` - Width of generated images (default: 768)
- `--height` - Height of generated images (default: 768)
- `--model` - ControlNet model to use (default: "canny")
- `--controlnet-module` - ControlNet module to use (default: "canny")
- `--controlnet-weight` - Weight of ControlNet influence (default: 0.8)
- `--sampler` - Sampler to use (default: "DPM++ 2M")
- `--scheduler` - Scheduler for the sampler (default: "Karras")
- `--steps` - Number of sampling steps (default: 30)
- `--cfg` - CFG scale for generation (default: 7.5)
- `--max-retries` - Maximum number of retries for failed operations (default: 3)
- `--retry-delay` - Delay between retries in milliseconds (default: 10000)
- `--batch-break` - Break duration between batches in milliseconds (default: 15000)

### Configuration File

The application can be configured using a YAML configuration file (`urasoe.config.yml`). Example:

```yaml
input_dir: "./my-images"
output_dir: "./results"
batch_size: 2
width: 512
height: 768
steps: 25
cfg: 7.0
model: "depth"
controlnet_module: "depth"
controlnet_weight: 0.9
sampler_name: "Euler a"
scheduler: "Karras"
checkpoint_model: "realisticVisionV51_v51VAE"
prompt: "karate master in dojo, high detail, realistic photography"
negative_prompt: "deformed, bad anatomy, disfigured, poorly drawn face"
max_retries: 5
retry_delay_ms: 15000
batch_break_ms: 20000
```

## Advanced Features

### Retry Mechanism

The application includes an intelligent retry mechanism for handling GPU memory issues. When an operation fails due to CUDA/GPU errors, the system will:

1. Wait for a configurable duration (increases with each retry)
2. Yield to the async runtime to help with memory management
3. Reattempt the operation
4. Provide detailed error reporting

### Batch Processing

To prevent GPU memory exhaustion when processing multiple images, the application:

1. Processes images in configurable batch sizes
2. Takes breaks between batches to allow GPU memory to clear
3. Reports detailed statistics on completion

## Requirements

- Rust (latest stable version)
- Stable Diffusion Automatic1111 running locally on port 7860
- ControlNet extension installed in Stable Diffusion
- `exiftool -ver`

## License

MIT
