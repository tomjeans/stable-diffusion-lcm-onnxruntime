#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <locale>
#include <codecvt>

class CLIPTokenizerComplete {
public:
    CLIPTokenizerComplete();
    
    // Initialize from vocab and merges files
    bool LoadFromFiles(const std::string& vocab_path, const std::string& merges_path);
    
    // Encode text to token IDs
    std::vector<int32_t> Encode(const std::string& text, bool add_special_tokens = true);
    
    // Decode token IDs to text
    std::string Decode(const std::vector<int32_t>& token_ids);
    
    int GetModelMaxLength() const { return model_max_length_; }
    
    // Check if tokenizer is properly initialized
    bool IsInitialized() const { return !vocab_.empty() && !merges_.empty(); }

private:
    std::unordered_map<std::string, int32_t> vocab_;
    std::unordered_map<int32_t, std::string> id_to_token_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::set<std::pair<std::string, std::string>> merges_set_;  // For fast lookup
    
    int32_t bos_token_id_;
    int32_t eos_token_id_;
    int32_t pad_token_id_;
    int32_t unk_token_id_;
    int model_max_length_;
    
    // Text preprocessing
    std::string NormalizeText(const std::string& text);
    std::vector<std::string> WhitespaceTokenize(const std::string& text);
    
    // BPE operations
    std::vector<std::string> BPE(const std::string& word);
    std::vector<int32_t> ConvertTokensToIds(const std::vector<std::string>& tokens);
    
    // File loading
    bool LoadVocab(const std::string& vocab_path);
    bool LoadMerges(const std::string& merges_path);
    
    // Simple JSON parsing (for vocab.json)
    std::unordered_map<std::string, int32_t> ParseVocabJSON(const std::string& json_content);
    
    // Helper: Get word pairs for BPE
    std::set<std::pair<std::string, std::string>> GetWordPairs(const std::vector<std::string>& word);
};
