# Stable Diffusion LCM C++ Implementation

A C++ implementation of Stable Diffusion 1.5 Latent Consistency Model (LCM) using ONNX Runtime and OpenCV. This implementation provides a native C++ alternative with similar functionality.

## Features

- **ONNX Runtime**: Fast inference using ONNX Runtime C++ API
- **OpenCV**: Image processing and saving
- **LCM Support**: Latent Consistency Model for fast image generation (4-8 steps)
- **CLIP Tokenizer**: Built-in text tokenization (simplified implementation)
- **LCM Scheduler**: Full LCM scheduler implementation
- **Windows Support**: Configured for Windows x64
- **Command-line Interface**: Compatible with the Python script's argument format

## Requirements

- Windows 10/11 (x64)
- CMake 3.10 or higher
- Visual Studio 2019 or later (with C++17 support)
- ONNX Runtime 1.23.2 (included in `onnxruntime-win-x64-1.23.2/`)
- OpenCV 4.x (included in `opencv/`)

## Model Files

You need to have the following ONNX model files organized in a model directory:

```
model_directory/
├── text_encoder/
│   └── model.onnx
├── unet/
│   └── model.onnx
└── vae_decoder/
    └── model.onnx
```

You can download these from the [Hugging Face repository](https://huggingface.co/happyme531/Stable-Diffusion-1.5-LCM-ONNX-RKNN2) or convert them from the original Stable Diffusion LCM model.

Optional: Each model directory can also contain a `config.json` file for model configuration (currently not fully parsed, but structure is in place).

## Tokenizer Files (Standalone C++ Mode)

For standalone C++ usage without Python dependencies, you need CLIP tokenizer files:

- **vocab.json**: CLIP tokenizer vocabulary file (maps tokens to IDs)
- **merges.txt**: BPE merge rules file

You can download these from:
- [Hugging Face CLIP tokenizer](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) - download `vocab.json` and `merges.txt`
- Or from any Stable Diffusion model repository that includes tokenizer files

Place these files in one of these locations (auto-detected):
- Current directory: `vocab.json` and `merges.txt`
- `tokenizer/` subdirectory: `tokenizer/vocab.json` and `tokenizer/merges.txt`
- Model directory: `model_directory/vocab.json` and `model_directory/merges.txt`

Or specify paths explicitly with `--vocab` and `--merges` command-line options.

## Building

1. Open a terminal in the project directory
2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```
3. Configure with CMake:
   ```bash
   cmake ..
   ```
4. Build:
   ```bash
   cmake --build . --config Release
   ```

The executable will be in `build/Release/sd.exe` (or `build/Debug/sd.exe` for Debug builds).

## Usage

### Command Line Arguments

The C++ version uses the same argument format as the Python script:

```bash
sd.exe --prompt TEXT -i MODEL_DIR -o OUTPUT_DIR [OPTIONS]
```

### Required Arguments

- `--prompt TEXT`: Text prompt for image generation
- `-i PATH`: Path to model directory containing text_encoder, unet, and vae_decoder subdirectories
- `-o PATH`: Output directory for generated images

### Optional Arguments

- `--vocab PATH`: Path to vocab.json file for CLIP tokenizer (optional, auto-detected if not specified)
- `--merges PATH`: Path to merges.txt file for CLIP tokenizer (optional, auto-detected if not specified)
- `--token-file PATH`: Pre-tokenized token file (deprecated, use --vocab and --merges instead)
- `--seed N`: Random seed for reproducibility (default: 93)
- `-s, --size WxH`: Image size in format WxH (default: 256x256)
- `--num-inference-steps N`: Number of inference steps (default: 4)
- `--guidance-scale F`: Guidance scale (default: 7.5)
- `--help, -h`: Show help message

### Examples

```bash
# Basic usage with default parameters (tokenizer files auto-detected)
sd.exe --prompt "a beautiful landscape" -i ./models -o ./output

# Specify tokenizer files explicitly
sd.exe --prompt "a beautiful landscape" -i ./models -o ./output --vocab vocab.json --merges merges.txt

# Custom size and steps
sd.exe --prompt "a futuristic city at night" -i ./models -o ./output -s 512x512 --num-inference-steps 8

# With custom seed and guidance scale
sd.exe --prompt "a cute cat sitting on the sofa" -i ./models -o ./output --seed 42 --guidance-scale 8.5 -s 384x384
```

### Output

The generated image will be saved in the output directory with a filename containing:
- Random seed
- Number of inference steps
- Sanitized prompt text
- Format: `seed_<seed>_steps<steps>_<prompt>.png`

## How It Works

1. **Text Tokenization**: The CLIP tokenizer converts the text prompt into token IDs
2. **Text Encoding**: Token IDs are passed through the text encoder ONNX model to get prompt embeddings
3. **Latent Initialization**: Random noise is generated in the latent space (scaled by scheduler's init_noise_sigma)
4. **LCM Inference Loop**: 
   - For each timestep, the UNet model predicts noise
   - The LCM scheduler steps the latents forward
   - Process repeats for the specified number of inference steps
5. **VAE Decoding**: The denoised latents are scaled and passed through the VAE decoder to generate the final image
6. **Post-processing**: Image is denormalized, converted from RGB to BGR, and saved using OpenCV

## Architecture

The implementation consists of several key components:

- **ONNXModel**: Wrapper class for loading and running ONNX models
- **LCMScheduler**: Implements the LCM diffusion scheduler with timestep scheduling
- **CLIPTokenizer**: Simplified CLIP tokenizer for text encoding (basic implementation)
- **StableDiffusionPipeline**: Main pipeline class that orchestrates the generation process

## LCM Scheduler

The implementation uses a full LCM scheduler:
- Generates beta and alpha schedules for diffusion
- Sets timesteps based on number of inference steps
- Performs denoising steps using LCM's simplified update rule
- Supports guidance scale embedding for classifier-free guidance

## Troubleshooting

### Model Files Not Found

Make sure the ONNX model files are in the correct directories:
- `text_encoder/model.onnx`
- `unet/model.onnx`
- `vae_decoder/model.onnx`

### DLL Errors

The ONNX Runtime DLLs should be copied automatically during build. If you get DLL errors, manually copy:
- `onnxruntime-win-x64-1.23.2/lib/onnxruntime.dll`
- `onnxruntime-win-x64-1.23.2/lib/onnxruntime_providers_shared.dll`

to the same directory as `sd.exe`.

### Tokenizer

The implementation includes two tokenizer options:

1. **Complete CLIP Tokenizer** (`CLIPTokenizerComplete`): Full BPE tokenization with vocab.json and merges.txt files. This is the recommended option for standalone C++ usage. It provides accurate tokenization matching the Python transformers library.

2. **Simplified Tokenizer** (`CLIPTokenizer`): Basic fallback tokenizer that works without external files but may produce lower quality results. Only use this if tokenizer files are not available.

The complete tokenizer will be used automatically if vocab.json and merges.txt files are found. Otherwise, the simplified tokenizer will be used with a warning.

### Model Input/Output Names

If you encounter errors about missing inputs or outputs, you may need to adjust the input/output names in the code to match your specific ONNX model exports. The code attempts to use standard names:
- Text encoder: `input_ids` → `last_hidden_state`
- UNet: `sample`, `timestep`, `encoder_hidden_states`, `timestep_cond` → `out_sample`
- VAE decoder: `latent_sample` → `sample`

## Performance

Typical performance on a modern CPU (varies by hardware):
- Text encoder: ~0.1-0.3s
- UNet (per step): ~1-5s (depends on resolution and model size)
- VAE decoder: ~0.5-2s
- Total (4 steps): ~5-20s

Performance can be improved by:
- Using GPU execution providers (requires ONNX Runtime with GPU support)
- Optimizing model quantization
- Using smaller image resolutions for faster iteration

## Comparison with Python Version

This C++ implementation is based on `run_onnx-lcm.py` and provides:

**Similarities:**
- Same command-line interface
- Same model structure and inference pipeline
- Compatible with the same ONNX models
- Similar output format

**Differences:**
- Native C++ performance (no Python overhead)
- Simplified CLIP tokenizer (Python version uses transformers library)
- No dependency on diffusers or transformers libraries
- Direct ONNX Runtime C++ API usage

## Project Structure

```
.
├── CMakeLists.txt              # CMake build configuration
├── main.cpp                    # Entry point with argument parsing
├── onnx_model.h/cpp           # ONNX model wrapper
├── lcm_scheduler.h/cpp        # LCM scheduler implementation
├── clip_tokenizer.h/cpp        # CLIP tokenizer (simplified)
├── stable_diffusion_pipeline.h/cpp  # Main pipeline class
├── utils.h                     # Utility functions
├── run_onnx-lcm.py            # Original Python reference implementation
└── README.md                   # This file
```

## License

This code is based on the Hugging Face repository and follows similar licensing terms.

## References

- [Stable Diffusion 1.5 LCM ONNX RKNN2](https://huggingface.co/happyme531/Stable-Diffusion-1.5-LCM-ONNX-RKNN2)
- [ONNX Runtime](https://onnxruntime.ai/)
- [OpenCV](https://opencv.org/)
- [Latent Consistency Models](https://arxiv.org/abs/2310.04378)


