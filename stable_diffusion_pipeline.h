#pragma once

#include "onnx_model.h"
#include "lcm_scheduler.h"
#include "clip_tokenizer.h"
#include "clip_tokenizer_complete.h"
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <opencv2/opencv.hpp>

class StableDiffusionPipeline {
public:
    StableDiffusionPipeline(
        const std::string& model_dir,
        const std::string& scheduler_config_path = "",
        const std::string& vocab_path = "",
        const std::string& merges_path = ""
    );
    
    cv::Mat Generate(
        const std::string& prompt,
        int height = 256,
        int width = 256,
        int num_inference_steps = 4,
        float guidance_scale = 8.5f,
        int seed = 42,
        const std::string& token_file = ""
    );

private:
    std::unique_ptr<ONNXModel> text_encoder_;
    std::unique_ptr<ONNXModel> unet_;
    std::unique_ptr<ONNXModel> vae_decoder_;
    std::unique_ptr<LCMScheduler> scheduler_;
    std::unique_ptr<CLIPTokenizer> tokenizer_;  // Keep for backward compatibility
    std::unique_ptr<CLIPTokenizerComplete> tokenizer_complete_;  // New complete tokenizer
    
    int vae_scale_factor_;
    int height_;
    int width_;
    
    std::pair<std::vector<float>, std::vector<int64_t>> EncodePrompt(const std::string& prompt, int num_images_per_prompt = 1);
    std::pair<std::vector<float>, std::vector<int64_t>> EncodePromptFromFile(const std::string& token_file, int num_images_per_prompt = 1);
    std::vector<float> PrepareLatents(int batch_size, int num_channels, int height, int width, int seed);
    std::vector<float> GetGuidanceScaleEmbedding(float w, int embedding_dim = 512);
    cv::Mat Postprocess(const std::vector<float>& image, int height, int width);
    
    Ort::Value CreateTensor(const std::vector<int64_t>& shape, const std::vector<float>& data);
    std::vector<float> TensorToVector(const Ort::Value& tensor);
};
