#include "stable_diffusion_pipeline.h"
#include <filesystem>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <fstream>
#include <limits>
#include "clip_tokenizer_complete.h"

StableDiffusionPipeline::StableDiffusionPipeline(
    const std::string& model_dir,
    const std::string& scheduler_config_path,
    const std::string& vocab_path,
    const std::string& merges_path) 
    : vae_scale_factor_(8),
      height_(256),
      width_(256) {
    
    std::filesystem::path base_path(model_dir);
    
    // Load models
    text_encoder_ = std::make_unique<ONNXModel>((base_path / "text_encoder").string());
    unet_ = std::make_unique<ONNXModel>((base_path / "unet").string());
    vae_decoder_ = std::make_unique<ONNXModel>((base_path / "vae_decoder").string());
    
    // Initialize scheduler
    scheduler_ = std::make_unique<LCMScheduler>();
    
    // Initialize tokenizer
    tokenizer_ = std::make_unique<CLIPTokenizer>();  // Keep for backward compatibility
    
    // Try to initialize complete tokenizer if vocab/merges paths provided
    tokenizer_complete_ = std::make_unique<CLIPTokenizerComplete>();
    
    std::string final_vocab_path = vocab_path;
    std::string final_merges_path = merges_path;
    
    // If paths not provided, try to find them in common locations
    if (final_vocab_path.empty()) {
        std::vector<std::string> possible_paths = {
            "vocab.json",
            "tokenizer/vocab.json",
            (base_path / "tokenizer" / "vocab.json").string(),
            (base_path / "vocab.json").string()
        };
        
        for (const auto& path : possible_paths) {
            if (std::filesystem::exists(path)) {
                final_vocab_path = path;
                break;
            }
        }
    }
    
    if (final_merges_path.empty()) {
        std::vector<std::string> possible_paths = {
            "merges.txt",
            "tokenizer/merges.txt",
            (base_path / "tokenizer" / "merges.txt").string(),
            (base_path / "merges.txt").string()
        };
        
        for (const auto& path : possible_paths) {
            if (std::filesystem::exists(path)) {
                final_merges_path = path;
                break;
            }
        }
    }
    
    // Load complete tokenizer if files found
    if (!final_vocab_path.empty() && !final_merges_path.empty()) {
        if (tokenizer_complete_->LoadFromFiles(final_vocab_path, final_merges_path)) {
            std::cout << "Loaded complete CLIP tokenizer from files" << std::endl;
        } else {
            std::cerr << "Warning: Failed to load complete tokenizer, using simplified version" << std::endl;
            tokenizer_complete_.reset();
        }
    } else {
        std::cout << "Tokenizer files not found, using simplified tokenizer" << std::endl;
        std::cout << "  Looking for vocab.json and merges.txt in:" << std::endl;
        std::cout << "    - Current directory" << std::endl;
        std::cout << "    - tokenizer/ subdirectory" << std::endl;
        std::cout << "    - Model directory" << std::endl;
        std::cout << "  Or specify paths with --vocab and --merges options" << std::endl;
        tokenizer_complete_.reset();
    }
    
    std::cout << "Pipeline initialized" << std::endl;
}

Ort::Value StableDiffusionPipeline::CreateTensor(const std::vector<int64_t>& shape, const std::vector<float>& data) {
    size_t total_size = 1;
    for (int64_t dim : shape) {
        total_size *= dim;
    }
    
    if (data.size() != total_size) {
        throw std::runtime_error("Data size mismatch");
    }
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    return Ort::Value::CreateTensor<float>(memory_info, 
                                          const_cast<float*>(data.data()), 
                                          data.size(),
                                          shape.data(), 
                                          shape.size());
}

std::vector<float> StableDiffusionPipeline::TensorToVector(const Ort::Value& tensor) {
    auto tensor_info = tensor.GetTensorTypeAndShapeInfo();
    size_t element_count = tensor_info.GetElementCount();
    
    const float* data = tensor.GetTensorData<float>();
    return std::vector<float>(data, data + element_count);
}

std::pair<std::vector<float>, std::vector<int64_t>> StableDiffusionPipeline::EncodePrompt(const std::string& prompt, int num_images_per_prompt) {
    // Tokenize - use complete tokenizer if available, otherwise fall back to simplified
    std::vector<int32_t> token_ids;
    
    if (tokenizer_complete_ && tokenizer_complete_->IsInitialized()) {
        token_ids = tokenizer_complete_->Encode(prompt);
        std::cout << "Using complete CLIP tokenizer" << std::endl;
    } else {
        token_ids = tokenizer_->Encode(prompt);
        std::cout << "Using simplified CLIP tokenizer (results may be poor)" << std::endl;
    }
    
    // Debug: Print token IDs to verify tokenization
    std::cout << "Token IDs (first 20): ";
    size_t print_count = std::min(static_cast<size_t>(20), token_ids.size());
    for (size_t i = 0; i < print_count; i++) {
        std::cout << token_ids[i] << " ";
    }
    std::cout << std::endl;
    
    // Check if tokenization looks valid (should have non-zero values, not all pad tokens)
    size_t non_zero_count = 0;
    for (auto id : token_ids) {
        if (id != 0) non_zero_count++;
    }
    std::cout << "Non-zero token IDs: " << non_zero_count << " / " << token_ids.size() << std::endl;
    
    if (non_zero_count < 5) {
        std::cerr << "WARNING: Tokenization appears invalid! Most tokens are zero (pad tokens)." << std::endl;
        std::cerr << "This will result in poor image quality. Please use --token-file option with a properly tokenized file." << std::endl;
    }
    
    // Create input tensor for text encoder
    std::vector<int64_t> input_shape = {1, static_cast<int64_t>(token_ids.size())};
    std::vector<int32_t> input_data(token_ids.begin(), token_ids.end());
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());
    
    // Run text encoder
    std::unordered_map<std::string, Ort::Value> inputs;
    inputs["input_ids"] = std::move(input_tensor);
    
    auto outputs = text_encoder_->Run(inputs);
    
    // Get the actual shape of the output
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    
    auto prompt_embeds = TensorToVector(outputs[0]);
    
    // Adjust shape for num_images_per_prompt
    if (num_images_per_prompt > 1 && shape.size() > 0) {
        shape[0] *= num_images_per_prompt;
        size_t original_size = prompt_embeds.size();
        prompt_embeds.resize(original_size * num_images_per_prompt);
        for (int i = 1; i < num_images_per_prompt; i++) {
            std::copy(prompt_embeds.begin(), 
                     prompt_embeds.begin() + original_size,
                     prompt_embeds.begin() + i * original_size);
        }
    }
    
    return {prompt_embeds, shape};
}

std::pair<std::vector<float>, std::vector<int64_t>> StableDiffusionPipeline::EncodePromptFromFile(const std::string& token_file, int num_images_per_prompt) {
    // Load token IDs from file (generated by tokenize_prompt.py)
    std::ifstream file(token_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open token file: " + token_file);
    }
    
    // Read as int32 array (77 tokens for CLIP)
    std::vector<int32_t> token_ids(77);
    file.read(reinterpret_cast<char*>(token_ids.data()), 77 * sizeof(int32_t));
    
    if (file.gcount() != 77 * sizeof(int32_t)) {
        throw std::runtime_error("Token file has incorrect size");
    }
    
    std::cout << "Loaded token IDs from file (first 10): ";
    for (size_t i = 0; i < 10; i++) {
        std::cout << token_ids[i] << " ";
    }
    std::cout << std::endl;
    
    // Create input tensor for text encoder
    std::vector<int64_t> input_shape = {1, 77};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<int32_t>(
        memory_info, token_ids.data(), token_ids.size(),
        input_shape.data(), input_shape.size());
    
    // Run text encoder
    std::unordered_map<std::string, Ort::Value> inputs;
    inputs["input_ids"] = std::move(input_tensor);
    
    auto outputs = text_encoder_->Run(inputs);
    
    // Get the actual shape of the output
    auto tensor_info = outputs[0].GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    
    auto prompt_embeds = TensorToVector(outputs[0]);
    
    // Adjust shape for num_images_per_prompt
    if (num_images_per_prompt > 1 && shape.size() > 0) {
        shape[0] *= num_images_per_prompt;
        size_t original_size = prompt_embeds.size();
        prompt_embeds.resize(original_size * num_images_per_prompt);
        for (int i = 1; i < num_images_per_prompt; i++) {
            std::copy(prompt_embeds.begin(), 
                     prompt_embeds.begin() + original_size,
                     prompt_embeds.begin() + i * original_size);
        }
    }
    
    return {prompt_embeds, shape};
}

std::vector<float> StableDiffusionPipeline::PrepareLatents(int batch_size, int num_channels, int height, int width, int seed) {
    int latent_height = height / vae_scale_factor_;
    int latent_width = width / vae_scale_factor_;
    
    size_t size = batch_size * num_channels * latent_height * latent_width;
    std::vector<float> latents(size);
    
    // Generate random noise
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < size; i++) {
        latents[i] = dist(gen) * scheduler_->GetInitNoiseSigma();
    }
    
    return latents;
}

std::vector<float> StableDiffusionPipeline::GetGuidanceScaleEmbedding(float w, int embedding_dim) {
    w = w * 1000.0f;
    int half_dim = embedding_dim / 2;
    
    float emb = std::log(10000.0f) / (half_dim - 1);
    
    std::vector<float> embedding(embedding_dim);
    for (int i = 0; i < half_dim; i++) {
        float val = std::exp(i * -emb);
        embedding[i] = std::sin(w * val);
        embedding[i + half_dim] = std::cos(w * val);
    }
    
    if (embedding_dim % 2 == 1) {
        embedding[embedding_dim - 1] = 0.0f;
    }
    
    return embedding;
}

cv::Mat StableDiffusionPipeline::Postprocess(const std::vector<float>& image, int height, int width) {
    // VAE output is [1, 3, H, W] = [batch, channels, height, width]
    // We need to extract the first batch item and transpose to [H, W, C]
    
    int channels = 3;
    int batch = 1;
    size_t expected_size = batch * channels * height * width;
    
    if (image.size() != expected_size) {
        std::cerr << "ERROR: VAE output size mismatch! Expected " << expected_size 
                  << " but got " << image.size() << std::endl;
        return cv::Mat::zeros(height, width, CV_8UC3);
    }
    
    cv::Mat img(height, width, CV_32FC3);
    
    // Extract from [B, C, H, W] format and transpose to [H, W, C]
    // Index calculation: batch_idx * C * H * W + channel_idx * H * W + y * W + x
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < channels; c++) {
                // For batch=0 (first batch item): 0 * C * H * W + c * H * W + y * W + x
                size_t idx = c * height * width + y * width + x;
                
                if (idx >= image.size()) {
                    std::cerr << "ERROR: Index out of bounds in postprocessing!" << std::endl;
                    return cv::Mat::zeros(height, width, CV_8UC3);
                }
                
                float val = image[idx];
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
                
                // Denormalize from [-1, 1] to [0, 1]
                val = std::clamp((val / 2.0f + 0.5f), 0.0f, 1.0f);
                img.at<cv::Vec3f>(y, x)[c] = val;
            }
        }
    }
    
    std::cout << "Postprocessing: VAE output range before denorm: [" << min_val << ", " << max_val << "]" << std::endl;
    
    // Check a few sample values after denormalization
    std::cout << "Postprocessing: Sample values after denorm: ";
    for (int i = 0; i < std::min(5, height * width); i++) {
        int y = i / width;
        int x = i % width;
        std::cout << "(" << img.at<cv::Vec3f>(y, x)[0] << "," << img.at<cv::Vec3f>(y, x)[1] << "," << img.at<cv::Vec3f>(y, x)[2] << ") ";
    }
    std::cout << std::endl;
    
    // Convert to BGR and uint8
    cv::Mat img_bgr;
    cv::cvtColor(img, img_bgr, cv::COLOR_RGB2BGR);
    img_bgr.convertTo(img_bgr, CV_8UC3, 255.0);
    
    // Check final image statistics
    cv::Scalar mean, stddev;
    cv::meanStdDev(img_bgr, mean, stddev);
    std::cout << "Final image stats: mean=" << mean[0] << ", stddev=" << stddev[0] << std::endl;
    
    return img_bgr;
}

cv::Mat StableDiffusionPipeline::Generate(
    const std::string& prompt,
    int height,
    int width,
    int num_inference_steps,
    float guidance_scale,
    int seed,
    const std::string& token_file) {
    
    height_ = height;
    width_ = width;
    
    std::cout << "Encoding prompt..." << std::endl;
    
    std::pair<std::vector<float>, std::vector<int64_t>> prompt_result;
    
    // Use token file if provided, otherwise try to find one, otherwise use tokenizer
    if (!token_file.empty() && std::filesystem::exists(token_file)) {
        std::cout << "Using provided token file: " << token_file << std::endl;
        prompt_result = EncodePromptFromFile(token_file, 1);
    } else {
        // Try to find a token file based on prompt
        std::string safe_prompt = prompt;
        std::replace(safe_prompt.begin(), safe_prompt.end(), ' ', '_');
        std::replace(safe_prompt.begin(), safe_prompt.end(), '/', '_');
        std::filesystem::path auto_token_file = std::filesystem::path("tokens") / (safe_prompt.substr(0, 50) + ".bin");
        
        if (std::filesystem::exists(auto_token_file)) {
            std::cout << "Using auto-detected token file: " << auto_token_file << std::endl;
            prompt_result = EncodePromptFromFile(auto_token_file.string(), 1);
        } else {
            std::cout << "WARNING: Using simplified tokenizer - results may be poor!" << std::endl;
            std::cout << "For better results, generate token file first:" << std::endl;
            std::cout << "  python tokenize_prompt.py --prompt \"" << prompt << "\" --output tokens/" << safe_prompt.substr(0, 50) << ".bin" << std::endl;
            prompt_result = EncodePrompt(prompt, 1);
        }
    }
    
    auto [prompt_embeds, prompt_shape] = prompt_result;
    
    // Debug: Check prompt embeddings
    std::cout << "Prompt embeddings shape: [";
    for (size_t i = 0; i < prompt_shape.size(); i++) {
        std::cout << prompt_shape[i];
        if (i < prompt_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Prompt embeddings size: " << prompt_embeds.size() << std::endl;
    if (!prompt_embeds.empty()) {
        std::cout << "Prompt embeddings range: [" 
                  << *std::min_element(prompt_embeds.begin(), prompt_embeds.end()) << ", "
                  << *std::max_element(prompt_embeds.begin(), prompt_embeds.end()) << "]" << std::endl;
    }
    
    int batch_size = 1;
    int num_images_per_prompt = 1;
    
    std::cout << "Preparing latents..." << std::endl;
    auto latents = PrepareLatents(batch_size * num_images_per_prompt, 4, height, width, seed);
    
    // Debug: Check initial latents range
    auto initial_latents_minmax = std::minmax_element(latents.begin(), latents.end());
    std::cout << "Initial latents range: [" 
              << *initial_latents_minmax.first << ", " << *initial_latents_minmax.second << "]" << std::endl;
    std::cout << "Python expected: [-3.822383, 3.539358]" << std::endl;
    if (latents.size() > 0) {
        float latents_mean = std::accumulate(latents.begin(), latents.end(), 0.0f) / latents.size();
        float latents_std = 0.0f;
        for (float val : latents) {
            latents_std += (val - latents_mean) * (val - latents_mean);
        }
        latents_std = std::sqrt(latents_std / latents.size());
        std::cout << "Initial latents mean: " << latents_mean << ", std: " << latents_std << std::endl;
        std::cout << "Python expected: mean=-0.000323, std=0.996669" << std::endl;
    }
    
    // Get the embedding dimension from UNet's timestep_cond input shape
    auto timestep_cond_shape = unet_->GetInputShape("timestep_cond");
    int embedding_dim = 256; // default
    if (timestep_cond_shape.size() >= 2) {
        embedding_dim = static_cast<int>(timestep_cond_shape[1]);
    }
    std::cout << "Using timestep_cond embedding dimension: " << embedding_dim << std::endl;
    
    // Get guidance scale embedding
    auto w_embedding = GetGuidanceScaleEmbedding(guidance_scale - 1.0f, embedding_dim);
    
    // Set timesteps
    scheduler_->SetTimesteps(num_inference_steps);
    scheduler_->ResetStepIndex();  // Reset step index before starting
    auto timesteps = scheduler_->GetTimesteps();
    
    std::cout << "Running denoising loop..." << std::endl;
    std::cout << "Timesteps: [";
    for (size_t i = 0; i < timesteps.size(); i++) {
        std::cout << timesteps[i];
        if (i < timesteps.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    int latent_height = height / vae_scale_factor_;
    int latent_width = width / vae_scale_factor_;
    
    std::vector<float> denoised_latents; // Store denoised from last step
    
    for (size_t i = 0; i < timesteps.size(); i++) {
        int t = timesteps[i];
        std::cout << "Step " << (i + 1) << "/" << timesteps.size() << " (timestep " << t << ")" << std::endl;
        
        // Prepare UNet inputs
        std::vector<int64_t> sample_shape = {batch_size, 4, latent_height, latent_width};
        auto sample_tensor = CreateTensor(sample_shape, latents);
        
        std::vector<int64_t> timestep_shape = {1};
        std::vector<int64_t> timestep_data = {static_cast<int64_t>(t)};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto timestep_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, timestep_data.data(), timestep_data.size(),
            timestep_shape.data(), timestep_shape.size());
        
        // Use the actual shape from text encoder output
        // If shape is different, we need to reshape
        std::vector<int64_t> encoder_hidden_states_shape = prompt_shape;
        if (encoder_hidden_states_shape.size() == 0) {
            // Fallback to default shape if not available
            encoder_hidden_states_shape = {1, 77, 768};
        }
        auto encoder_hidden_states_tensor = CreateTensor(encoder_hidden_states_shape, prompt_embeds);
        
        std::vector<int64_t> timestep_cond_shape_vec = {1, static_cast<int64_t>(embedding_dim)};
        auto timestep_cond_tensor = CreateTensor(timestep_cond_shape_vec, w_embedding);
        
        std::unordered_map<std::string, Ort::Value> unet_inputs;
        unet_inputs["sample"] = std::move(sample_tensor);
        unet_inputs["timestep"] = std::move(timestep_tensor);
        unet_inputs["encoder_hidden_states"] = std::move(encoder_hidden_states_tensor);
        unet_inputs["timestep_cond"] = std::move(timestep_cond_tensor);
        
        // Run UNet
        auto unet_outputs = unet_->Run(unet_inputs);
        auto noise_pred = TensorToVector(unet_outputs[0]);
        
        // Debug: Check noise prediction range
        if (i == 0 || i == timesteps.size() - 1 || i == timesteps.size() - 2) {
            auto minmax = std::minmax_element(noise_pred.begin(), noise_pred.end());
            std::cout << "  Noise pred range at step " << (i+1) << ": [" 
                      << *minmax.first << ", " << *minmax.second << "]" << std::endl;
            // Compare with Python: Step 1 should be ~[-3.8, 3.5], Step 4 should be ~[-2.38, 2.45]
        }
        
        // Scheduler step (pass step index)
        auto [prev_sample, denoised] = scheduler_->Step(noise_pred, t, latents, static_cast<int>(i));
        latents = prev_sample;
        
        // Debug: Check latents range after step
        if (i == 0 || i == timesteps.size() - 1 || i == timesteps.size() - 2) {
            auto minmax = std::minmax_element(latents.begin(), latents.end());
            std::cout << "  Latents range after step " << (i+1) << ": [" 
                      << *minmax.first << ", " << *minmax.second << "]" << std::endl;
            minmax = std::minmax_element(denoised.begin(), denoised.end());
            std::cout << "  Denoised range after step " << (i+1) << ": [" 
                      << *minmax.first << ", " << *minmax.second << "]" << std::endl;
        }
        
        // Store denoised from each step
        // The Python code uses denoised from the LAST scheduler.step() call
        // So we always update denoised_latents to the latest denoised value
        denoised_latents = denoised;
        
        // Debug: Compare denoised vs latents at last step
        if (i == timesteps.size() - 1) {
            float denoised_mean = std::accumulate(denoised.begin(), denoised.end(), 0.0f) / denoised.size();
            float latents_mean = std::accumulate(latents.begin(), latents.end(), 0.0f) / latents.size();
            std::cout << "  Final step (t=" << t << "): denoised_mean=" << denoised_mean 
                      << ", latents_mean=" << latents_mean << std::endl;
            std::cout << "  Using denoised from final step (t=" << t << ") as per Python code" << std::endl;
        }
    }
    
    std::cout << "Decoding with VAE..." << std::endl;
    
    // The Python code uses denoised from the last scheduler.step() call
    // At the final timestep (259), Python shows denoised equals latents
    // So we should use denoised from the last step
    std::vector<float> vae_input;
    
    if (!denoised_latents.empty()) {
        vae_input = denoised_latents;
        std::cout << "Using denoised from final step (t=259) for VAE, as per Python code" << std::endl;
        std::cout << "Denoised range: [" 
                  << *std::min_element(vae_input.begin(), vae_input.end()) << ", "
                  << *std::max_element(vae_input.begin(), vae_input.end()) << "]" << std::endl;
        std::cout << "Python expected: [-2.378411, 2.596775]" << std::endl;
    } else {
        vae_input = latents;
        std::cout << "WARNING: denoised_latents empty, using latents instead" << std::endl;
    }
    
    // Show latents for comparison
    auto latents_minmax = std::minmax_element(latents.begin(), latents.end());
    std::cout << "For comparison, latents range: [" 
              << *latents_minmax.first << ", " << *latents_minmax.second << "]" << std::endl;
    
    // Scale latents - try to get from config, otherwise use default
    // The Python code does: denoised /= self.vae_decoder.config["scaling_factor"]
    // Typical scaling factor is 0.18215 for SD 1.x models
    float scaling_factor = 0.18215f; // Default VAE scaling factor
    // TODO: Get from vae_decoder config if available
    
    std::cout << "Using VAE scaling factor: " << scaling_factor << std::endl;
    std::cout << "VAE input range before scaling: [" 
              << *std::min_element(vae_input.begin(), vae_input.end()) << ", "
              << *std::max_element(vae_input.begin(), vae_input.end()) << "]" << std::endl;
    
    // Scale the VAE input by the VAE scaling factor
    // This converts from the model's latent space to VAE's expected input space
    // Note: We divide by scaling_factor (Python: denoised /= scaling_factor)
    for (auto& val : vae_input) {
        val /= scaling_factor;
    }
    
    std::cout << "VAE input range after scaling: [" 
              << *std::min_element(vae_input.begin(), vae_input.end()) << ", "
              << *std::max_element(vae_input.begin(), vae_input.end()) << "]" << std::endl;
    
    // Debug: Check if scaled values are reasonable
    // VAE typically expects values in a certain range
    size_t out_of_range_count = 0;
    for (const auto& val : vae_input) {
        if (std::abs(val) > 20.0f) {
            out_of_range_count++;
        }
    }
    if (out_of_range_count > 0) {
        std::cout << "WARNING: " << out_of_range_count << " values are outside typical VAE input range [-20, 20]" << std::endl;
    }
    
    // Prepare VAE input tensor
    std::vector<int64_t> vae_input_shape = {1, 4, latent_height, latent_width};
    auto vae_input_tensor = CreateTensor(vae_input_shape, vae_input);
    
    std::unordered_map<std::string, Ort::Value> vae_inputs;
    vae_inputs["latent_sample"] = std::move(vae_input_tensor);
    
    auto vae_outputs = vae_decoder_->Run(vae_inputs);
    
    // Get VAE output shape
    auto vae_tensor_info = vae_outputs[0].GetTensorTypeAndShapeInfo();
    auto vae_output_shape = vae_tensor_info.GetShape();
    std::cout << "VAE output shape: [";
    for (size_t i = 0; i < vae_output_shape.size(); i++) {
        std::cout << vae_output_shape[i];
        if (i < vae_output_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    auto image = TensorToVector(vae_outputs[0]);
    
    std::cout << "VAE output size: " << image.size() << std::endl;
    if (!image.empty()) {
        std::cout << "VAE output range: [" 
                  << *std::min_element(image.begin(), image.end()) << ", "
                  << *std::max_element(image.begin(), image.end()) << "]" << std::endl;
    }
    
    std::cout << "Postprocessing..." << std::endl;
    return Postprocess(image, height, width);
}
