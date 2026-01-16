#include "onnx_model.h"
#include <filesystem>
#include <iostream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

ONNXModel::ONNXModel(const std::string& model_dir) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "ONNXModel"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    
    std::filesystem::path model_path = std::filesystem::path(model_dir) / "model.onnx";
    std::filesystem::path config_path = std::filesystem::path(model_dir) / "config.json";

    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path.string());
    }

    // Load config (simplified - just store as map for now)
    // In a full implementation, you'd parse JSON properly
    if (std::filesystem::exists(config_path)) {
        // For now, we'll skip JSON parsing - can be added later if needed
        // config_ will remain empty
    }

    // Create session
    Ort::SessionOptions session_options;
    
#ifdef _WIN32
    // Use Windows API for path conversion instead of deprecated codecvt
    std::string model_path_str = model_path.string();
    int wchars_num = MultiByteToWideChar(CP_UTF8, 0, model_path_str.c_str(), -1, NULL, 0);
    std::wstring w_model_path(wchars_num, 0);
    MultiByteToWideChar(CP_UTF8, 0, model_path_str.c_str(), -1, &w_model_path[0], wchars_num);
    session_ = std::make_unique<Ort::Session>(env_, w_model_path.c_str(), session_options);
#else
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
#endif

    // Get input/output names and shapes
    size_t num_input_nodes = session_->GetInputCount();
    size_t num_output_nodes = session_->GetOutputCount();

    for (size_t i = 0; i < num_input_nodes; i++) {
        auto input_name = session_->GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        input_names_.push_back(input_name.get());
        
        auto input_type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        input_shapes_.push_back(shape);
    }

    for (size_t i = 0; i < num_output_nodes; i++) {
        auto output_name = session_->GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
        output_names_.push_back(output_name.get());
        
        auto output_type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        output_shapes_.push_back(shape);
    }

    model_name_ = std::filesystem::path(model_dir).filename().string();
    std::cout << "Loaded model: " << model_name_ << std::endl;
}

std::vector<Ort::Value> ONNXModel::Run(std::unordered_map<std::string, Ort::Value>& inputs) {
    std::vector<const char*> input_names;
    std::vector<Ort::Value> input_values;
    
    for (const auto& name : input_names_) {
        auto it = inputs.find(name);
        if (it == inputs.end()) {
            throw std::runtime_error("Missing input: " + name);
        }
        input_names.push_back(name.c_str());
        input_values.push_back(std::move(it->second));
    }

    std::vector<const char*> output_names;
    for (const auto& name : output_names_) {
        output_names.push_back(name.c_str());
    }

    auto outputs = session_->Run(Ort::RunOptions{nullptr}, 
                                 input_names.data(), input_values.data(), input_names.size(),
                                 output_names.data(), output_names.size());

    return outputs;
}

std::vector<int64_t> ONNXModel::GetInputShape(const std::string& input_name) const {
    for (size_t i = 0; i < input_names_.size(); i++) {
        if (input_names_[i] == input_name) {
            return input_shapes_[i];
        }
    }
    throw std::runtime_error("Input name not found: " + input_name);
}
