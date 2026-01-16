#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>

class CLIPTokenizer {
public:
    CLIPTokenizer();
    CLIPTokenizer(const std::string& vocab_path, const std::string& merges_path);
    
    std::vector<int32_t> Encode(const std::string& text, bool add_special_tokens = true);
    std::string Decode(const std::vector<int32_t>& token_ids);
    
    int GetModelMaxLength() const { return model_max_length_; }

private:
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    int32_t bos_token_id_;
    int32_t eos_token_id_;
    int32_t pad_token_id_;
    int model_max_length_;
    
    std::vector<std::string> BasicTokenize(const std::string& text);
    std::vector<int32_t> ApplyBPE(const std::vector<std::string>& tokens);
    void LoadVocab(const std::string& vocab_path);
    void LoadMerges(const std::string& merges_path);
};
