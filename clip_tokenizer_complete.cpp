#include "clip_tokenizer_complete.h"
#include <filesystem>
#include <iostream>
#include <regex>
#include <iomanip>

CLIPTokenizerComplete::CLIPTokenizerComplete()
    : bos_token_id_(49406),
      eos_token_id_(49407),
      pad_token_id_(0),
      unk_token_id_(0),
      model_max_length_(77) {
}

bool CLIPTokenizerComplete::LoadFromFiles(const std::string& vocab_path, const std::string& merges_path) {
    bool vocab_loaded = LoadVocab(vocab_path);
    bool merges_loaded = LoadMerges(merges_path);
    
    if (!vocab_loaded) {
        std::cerr << "Warning: Failed to load vocab file: " << vocab_path << std::endl;
    }
    if (!merges_loaded) {
        std::cerr << "Warning: Failed to load merges file: " << merges_path << std::endl;
    }
    
    // Try to find unk_token_id from vocab
    auto unk_it = vocab_.find("<|endoftext|>");
    if (unk_it != vocab_.end()) {
        unk_token_id_ = unk_it->second;
    }
    
    return vocab_loaded && merges_loaded;
}

bool CLIPTokenizerComplete::LoadVocab(const std::string& vocab_path) {
    if (!std::filesystem::exists(vocab_path)) {
        std::cerr << "Vocab file does not exist: " << vocab_path << std::endl;
        return false;
    }
    
    std::ifstream file(vocab_path);
    if (!file.is_open()) {
        std::cerr << "Could not open vocab file: " << vocab_path << std::endl;
        return false;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    file.close();
    
    vocab_ = ParseVocabJSON(json_content);
    
    // Build reverse mapping
    id_to_token_.clear();
    for (const auto& [token, id] : vocab_) {
        id_to_token_[id] = token;
    }
    
    std::cout << "Loaded " << vocab_.size() << " tokens from vocab file" << std::endl;
    return !vocab_.empty();
}

std::unordered_map<std::string, int32_t> CLIPTokenizerComplete::ParseVocabJSON(const std::string& json_content) {
    std::unordered_map<std::string, int32_t> vocab;
    
    // Simple JSON parser for vocab.json
    // Format: {"token1": id1, "token2": id2, ...}
    // More robust regex that handles escaped quotes and unicode
    // Use a delimiter for raw string to avoid issues with quotes
    std::regex token_regex(R"xxx("((?:[^"\\]|\\.|\\u[0-9a-fA-F]{4})*)":\s*(\d+))xxx");
    std::sregex_iterator iter(json_content.begin(), json_content.end(), token_regex);
    std::sregex_iterator end;
    
    for (; iter != end; ++iter) {
        std::smatch match = *iter;
        std::string token = match[1].str();
        int32_t id = std::stoi(match[2].str());
        
        // Unescape JSON string
        std::string unescaped_token;
        unescaped_token.reserve(token.length());
        
        for (size_t i = 0; i < token.length(); ++i) {
            if (token[i] == '\\' && i + 1 < token.length()) {
                switch (token[i + 1]) {
                    case 'n': unescaped_token += '\n'; ++i; break;
                    case 't': unescaped_token += '\t'; ++i; break;
                    case 'r': unescaped_token += '\r'; ++i; break;
                    case '\\': unescaped_token += '\\'; ++i; break;
                    case '"': unescaped_token += '"'; ++i; break;
                    case '/': unescaped_token += '/'; ++i; break;
                    case 'b': unescaped_token += '\b'; ++i; break;
                    case 'f': unescaped_token += '\f'; ++i; break;
                    case 'u': 
                        // Unicode escape: \uXXXX
                        if (i + 5 < token.length()) {
                            try {
                                std::string hex = token.substr(i + 2, 4);
                                unsigned int code = std::stoul(hex, nullptr, 16);
                                // Handle UTF-16 surrogate pairs (simplified - for most cases)
                                if (code >= 0xD800 && code <= 0xDBFF) {
                                    // High surrogate - skip for now (would need low surrogate)
                                    i += 5;
                                    continue;
                                }
                                // Convert to UTF-8 (simplified for ASCII range)
                                if (code < 0x80) {
                                    unescaped_token += static_cast<char>(code);
                                } else if (code < 0x800) {
                                    unescaped_token += static_cast<char>(0xC0 | (code >> 6));
                                    unescaped_token += static_cast<char>(0x80 | (code & 0x3F));
                                } else {
                                    unescaped_token += static_cast<char>(0xE0 | (code >> 12));
                                    unescaped_token += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                                    unescaped_token += static_cast<char>(0x80 | (code & 0x3F));
                                }
                                i += 5;
                            } catch (...) {
                                // Invalid unicode, skip
                                unescaped_token += token[i];
                            }
                        } else {
                            unescaped_token += token[i];
                        }
                        break;
                    default: unescaped_token += token[i]; break;
                }
            } else {
                unescaped_token += token[i];
            }
        }
        
        vocab[unescaped_token] = id;
    }
    
    return vocab;
}

bool CLIPTokenizerComplete::LoadMerges(const std::string& merges_path) {
    if (!std::filesystem::exists(merges_path)) {
        std::cerr << "Merges file does not exist: " << merges_path << std::endl;
        return false;
    }
    
    std::ifstream file(merges_path);
    if (!file.is_open()) {
        std::cerr << "Could not open merges file: " << merges_path << std::endl;
        return false;
    }
    
    merges_.clear();
    merges_set_.clear();
    
    std::string line;
    bool first_line = true;
    
    while (std::getline(file, line)) {
        // Skip first line (version info) and empty lines
        if (first_line) {
            first_line = false;
            continue;
        }
        
        if (line.empty()) continue;
        
        // Parse merge pair: "token1 token2"
        std::istringstream iss(line);
        std::string token1, token2;
        if (iss >> token1 >> token2) {
            merges_.emplace_back(token1, token2);
            merges_set_.emplace(token1, token2);
        }
    }
    
    std::cout << "Loaded " << merges_.size() << " BPE merge rules" << std::endl;
    return !merges_.empty();
}

std::string CLIPTokenizerComplete::NormalizeText(const std::string& text) {
    std::string normalized;
    normalized.reserve(text.length());
    
    for (char c : text) {
        // Convert to lowercase
        char lower = std::tolower(c, std::locale());
        
        // Replace whitespace with space
        if (std::isspace(lower)) {
            normalized += ' ';
        } else {
            normalized += lower;
        }
    }
    
    // Trim and normalize whitespace
    std::string result;
    bool prev_space = false;
    for (char c : normalized) {
        if (c == ' ') {
            if (!prev_space) {
                result += ' ';
                prev_space = true;
            }
        } else {
            result += c;
            prev_space = false;
        }
    }
    
    // Trim leading/trailing spaces
    if (!result.empty() && result[0] == ' ') {
        result = result.substr(1);
    }
    if (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    
    return result;
}

std::vector<std::string> CLIPTokenizerComplete::WhitespaceTokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream iss(text);
    std::string token;
    
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    return tokens;
}

std::set<std::pair<std::string, std::string>> CLIPTokenizerComplete::GetWordPairs(const std::vector<std::string>& word) {
    std::set<std::pair<std::string, std::string>> pairs;
    
    if (word.size() < 2) {
        return pairs;
    }
    
    for (size_t i = 0; i < word.size() - 1; ++i) {
        pairs.emplace(word[i], word[i + 1]);
    }
    
    return pairs;
}

std::vector<std::string> CLIPTokenizerComplete::BPE(const std::string& word) {
    if (word.empty()) {
        return {};
    }
    
    // Split word into characters
    std::vector<std::string> word_tokens;
    for (char c : word) {
        word_tokens.push_back(std::string(1, c));
    }
    
    // Add end-of-word marker
    word_tokens.push_back("</w>");
    
    // Apply BPE merges
    while (word_tokens.size() > 1) {
        auto pairs = GetWordPairs(word_tokens);
        
        // Find the highest priority merge (first in merges_ list)
        std::pair<std::string, std::string> best_pair;
        bool found = false;
        
        for (const auto& merge : merges_) {
            if (pairs.find(merge) != pairs.end()) {
                best_pair = merge;
                found = true;
                break;
            }
        }
        
        if (!found) {
            break;  // No more merges possible
        }
        
        // Merge the pair
        std::vector<std::string> new_word;
        size_t i = 0;
        
        while (i < word_tokens.size()) {
            if (i < word_tokens.size() - 1 &&
                word_tokens[i] == best_pair.first &&
                word_tokens[i + 1] == best_pair.second) {
                // Merge
                new_word.push_back(best_pair.first + best_pair.second);
                i += 2;
            } else {
                new_word.push_back(word_tokens[i]);
                i += 1;
            }
        }
        
        word_tokens = new_word;
    }
    
    // Remove </w> markers
    for (auto& token : word_tokens) {
        if (token == "</w>") {
            token = "";
        } else if (token.size() > 4 && token.substr(token.size() - 4) == "</w>") {
            token = token.substr(0, token.size() - 4);
        }
    }
    
    // Remove empty tokens
    word_tokens.erase(
        std::remove_if(word_tokens.begin(), word_tokens.end(),
                      [](const std::string& s) { return s.empty(); }),
        word_tokens.end());
    
    return word_tokens;
}

std::vector<int32_t> CLIPTokenizerComplete::ConvertTokensToIds(const std::vector<std::string>& tokens) {
    std::vector<int32_t> token_ids;
    token_ids.reserve(tokens.size());
    
    for (const auto& token : tokens) {
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            token_ids.push_back(it->second);
        } else {
            // Try to find token with </w> suffix
            std::string token_with_suffix = token + "</w>";
            auto it2 = vocab_.find(token_with_suffix);
            if (it2 != vocab_.end()) {
                token_ids.push_back(it2->second);
            } else {
                // Use UNK token
                token_ids.push_back(unk_token_id_);
            }
        }
    }
    
    return token_ids;
}

std::vector<int32_t> CLIPTokenizerComplete::Encode(const std::string& text, bool add_special_tokens) {
    std::vector<int32_t> token_ids;
    
    if (add_special_tokens) {
        token_ids.push_back(bos_token_id_);
    }
    
    // Normalize text
    std::string normalized = NormalizeText(text);
    
    // Whitespace tokenize
    auto words = WhitespaceTokenize(normalized);
    
    // Apply BPE to each word
    std::vector<std::string> all_tokens;
    for (const auto& word : words) {
        auto bpe_tokens = BPE(word);
        all_tokens.insert(all_tokens.end(), bpe_tokens.begin(), bpe_tokens.end());
    }
    
    // Convert tokens to IDs
    auto word_token_ids = ConvertTokensToIds(all_tokens);
    token_ids.insert(token_ids.end(), word_token_ids.begin(), word_token_ids.end());
    
    if (add_special_tokens) {
        token_ids.push_back(eos_token_id_);
    }
    
    // Pad or truncate to model_max_length_
    if (token_ids.size() < model_max_length_) {
        token_ids.resize(model_max_length_, pad_token_id_);
    } else if (token_ids.size() > model_max_length_) {
        token_ids.resize(model_max_length_);
        if (add_special_tokens && token_ids.size() > 0) {
            token_ids[model_max_length_ - 1] = eos_token_id_;
        }
    }
    
    return token_ids;
}

std::string CLIPTokenizerComplete::Decode(const std::vector<int32_t>& token_ids) {
    std::string text;
    
    for (int32_t id : token_ids) {
        if (id == pad_token_id_ || id == bos_token_id_ || id == eos_token_id_) {
            continue;
        }
        
        auto it = id_to_token_.find(id);
        if (it != id_to_token_.end()) {
            std::string token = it->second;
            
            // Remove </w> markers
            if (token.size() > 4 && token.substr(token.size() - 4) == "</w>") {
                token = token.substr(0, token.size() - 4);
            }
            
            text += token;
        }
    }
    
    return text;
}
