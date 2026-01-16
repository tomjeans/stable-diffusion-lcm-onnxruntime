#include "clip_tokenizer.h"
#include <filesystem>
#include <iostream>
#include <cctype>

CLIPTokenizer::CLIPTokenizer() 
    : bos_token_id_(49406),
      eos_token_id_(49407),
      pad_token_id_(0),
      model_max_length_(77) {
    // Default CLIP tokenizer - simplified version
    // In a real implementation, you'd load the vocab and merges files
    // For now, we'll create a basic tokenizer
}

CLIPTokenizer::CLIPTokenizer(const std::string& vocab_path, const std::string& merges_path)
    : CLIPTokenizer() {
    if (std::filesystem::exists(vocab_path)) {
        LoadVocab(vocab_path);
    }
    if (std::filesystem::exists(merges_path)) {
        LoadMerges(merges_path);
    }
}

void CLIPTokenizer::LoadVocab(const std::string& vocab_path) {
    // Simplified vocab loading - in production, parse JSON properly
    // For now, we'll use a basic tokenizer without full vocab
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open vocab file: " << vocab_path << std::endl;
        return;
    }
    // TODO: Implement proper JSON parsing for vocab
}

void CLIPTokenizer::LoadMerges(const std::string& merges_path) {
    std::ifstream file(merges_path);
    std::string line;
    
    // Skip first line (version info)
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string token1, token2;
        if (iss >> token1 >> token2) {
            merges_.emplace_back(token1, token2);
        }
    }
}

std::vector<std::string> CLIPTokenizer::BasicTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string current_token;
    
    for (char c : text) {
        if (std::isspace(c)) {
            if (!current_token.empty()) {
                tokens.push_back(current_token);
                current_token.clear();
            }
        } else {
            current_token += c;
        }
    }
    
    if (!current_token.empty()) {
        tokens.push_back(current_token);
    }
    
    return tokens;
}

std::vector<int32_t> CLIPTokenizer::ApplyBPE(const std::vector<std::string>& tokens) {
    // Simplified BPE - in real implementation, apply merge rules
    std::vector<int32_t> token_ids;
    
    for (const auto& token : tokens) {
        // Try to find token in vocab
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            token_ids.push_back(it->second);
        } else {
            // Fallback: use UNK token or split into characters
            for (char c : token) {
                std::string char_str(1, c);
                auto char_it = vocab_.find(char_str);
                if (char_it != vocab_.end()) {
                    token_ids.push_back(char_it->second);
                }
            }
        }
    }
    
    return token_ids;
}

std::vector<int32_t> CLIPTokenizer::Encode(const std::string& text, bool add_special_tokens) {
    std::vector<int32_t> token_ids;
    
    if (add_special_tokens) {
        token_ids.push_back(bos_token_id_);
    }
    
    // Basic tokenization
    auto tokens = BasicTokenize(text);
    
    // Apply BPE (simplified)
    auto bpe_tokens = ApplyBPE(tokens);
    token_ids.insert(token_ids.end(), bpe_tokens.begin(), bpe_tokens.end());
    
    if (add_special_tokens) {
        token_ids.push_back(eos_token_id_);
    }
    
    // Pad or truncate to model_max_length_
    if (token_ids.size() < model_max_length_) {
        token_ids.resize(model_max_length_, pad_token_id_);
    } else if (token_ids.size() > model_max_length_) {
        token_ids.resize(model_max_length_);
        token_ids[model_max_length_ - 1] = eos_token_id_;
    }
    
    return token_ids;
}

std::string CLIPTokenizer::Decode(const std::vector<int32_t>& token_ids) {
    std::string text;
    
    for (int32_t id : token_ids) {
        if (id == pad_token_id_ || id == bos_token_id_ || id == eos_token_id_) {
            continue;
        }
        
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            text += it->second;
        }
    }
    
    return text;
}
