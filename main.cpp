#include "stable_diffusion_pipeline.h"
#include <iostream>
#include <string>
#include <filesystem>

#ifdef _WIN32
#include <windows.h>
#endif

void PrintUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  --prompt TEXT          Text prompt for image generation (required)\n"
              << "  -i PATH                Path to model directory (required)\n"
              << "  -o PATH                Output directory (required)\n"
              << "  --token-file PATH      Pre-tokenized token file (optional, deprecated)\n"
              << "  --vocab PATH           Path to vocab.json file for CLIP tokenizer (optional)\n"
              << "  --merges PATH          Path to merges.txt file for CLIP tokenizer (optional)\n"
              << "  --seed N               Random seed (default: 93)\n"
              << "  -s, --size WxH         Image size (default: 256x256)\n"
              << "  --num-inference-steps N Number of inference steps (default: 4)\n"
              << "  --guidance-scale F     Guidance scale (default: 7.5)\n"
              << "\nNote: For standalone C++ usage, provide --vocab and --merges options.\n"
              << "  The tokenizer will auto-detect files in current dir, tokenizer/, or model dir.\n"
              << std::endl;
}

int ParseSize(const std::string& size_str, int& width, int& height) {
    size_t x_pos = size_str.find('x');
    if (x_pos == std::string::npos) {
        return -1;
    }
    
    try {
        width = std::stoi(size_str.substr(0, x_pos));
        height = std::stoi(size_str.substr(x_pos + 1));
        return 0;
    } catch (...) {
        return -1;
    }
}

int main(int argc, char* argv[]) {
    std::string prompt;
    std::string model_dir;
    std::string output_dir;
    std::string token_file;
    std::string vocab_path;
    std::string merges_path;
    int seed = 93;
    std::string size_str = "256x256";
    int num_inference_steps = 4;
    float guidance_scale = 7.5f;
    
    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "-i" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--size") && i + 1 < argc) {
            size_str = argv[++i];
        } else if (arg == "--num-inference-steps" && i + 1 < argc) {
            num_inference_steps = std::stoi(argv[++i]);
        } else if (arg == "--guidance-scale" && i + 1 < argc) {
            guidance_scale = std::stof(argv[++i]);
        } else if (arg == "--token-file" && i + 1 < argc) {
            token_file = argv[++i];
        } else if (arg == "--vocab" && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (arg == "--merges" && i + 1 < argc) {
            merges_path = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        }
    }
    
    // Validate required arguments
    if (prompt.empty() || model_dir.empty() || output_dir.empty()) {
        std::cerr << "Error: --prompt, -i, and -o are required" << std::endl;
        PrintUsage(argv[0]);
        return 1;
    }
    
    // Parse size
    int width, height;
    if (ParseSize(size_str, width, height) != 0) {
        std::cerr << "Error: Invalid size format. Use WxH (e.g., 256x256)" << std::endl;
        return 1;
    }
    
    // Validate model directory
    if (!std::filesystem::exists(model_dir)) {
        std::cerr << "Error: Model directory does not exist: " << model_dir << std::endl;
        return 1;
    }
    
    // Create output directory
    std::filesystem::create_directories(output_dir);
    
    try {
        std::cout << "Initializing pipeline..." << std::endl;
        std::cout << "Model directory: " << model_dir << std::endl;
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Size: " << width << "x" << height << std::endl;
        std::cout << "Seed: " << seed << std::endl;
        std::cout << "Inference steps: " << num_inference_steps << std::endl;
        std::cout << "Guidance scale: " << guidance_scale << std::endl;
        
        StableDiffusionPipeline pipeline(model_dir, "", vocab_path, merges_path);
        
        std::cout << "\nGenerating image..." << std::endl;
        cv::Mat image = pipeline.Generate(
            prompt,
            height,
            width,
            num_inference_steps,
            guidance_scale,
            seed,
            token_file
        );
        
        // Generate output filename
        std::string safe_prompt = prompt;
        std::replace(safe_prompt.begin(), safe_prompt.end(), ' ', '_');
        std::replace(safe_prompt.begin(), safe_prompt.end(), '/', '_');
        
        std::string filename = "seed_" + std::to_string(seed) + 
                              "_steps" + std::to_string(num_inference_steps) + 
                              "_" + "_" + ".png";
        
        std::filesystem::path output_path = std::filesystem::path(output_dir) / filename;
        
        std::cout << "Saving image to: " << output_path << std::endl;
        cv::imwrite(output_path.string(), image);
        
        std::cout << "Done!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
