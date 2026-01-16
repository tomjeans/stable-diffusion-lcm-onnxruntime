#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <map>

class ONNXModel {
public:
    ONNXModel(const std::string& model_dir);
    ~ONNXModel() = default;

    std::vector<Ort::Value> Run(std::unordered_map<std::string, Ort::Value>& inputs);

    std::map<std::string, std::string> GetConfig() const { return config_; }
    std::string GetModelName() const { return model_name_; }
    std::vector<int64_t> GetInputShape(const std::string& input_name) const;

private:
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    Ort::MemoryInfo memory_info_;
    std::map<std::string, std::string> config_;
    std::string model_name_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
};
