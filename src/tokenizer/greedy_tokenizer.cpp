#include "greedy_tokenizer.hpp"

GreedyTokenizer::GreedyTokenizer() {
    // TODO: compare with memory mapped tokenizer
    std::ifstream ifs(this->token_filepath);
    if (!ifs) {
        throw std::runtime_error("Failed to open token file: " + this->token_filepath);
    }
    
    std::string line;
    while (std::getline(ifs, line)) {
        // Find the last space (separator between base64 token and ID)
        size_t space_pos = line.rfind(' ');
        if (space_pos == std::string::npos) {
            continue; // Skip malformed lines
        }
        
        std::string base64_token = line.substr(0, space_pos);
        int token_id = std::stoi(line.substr(space_pos + 1));
        
        // Decode the base64 token into bytestrings (instead of 6 bits per char, use 8 bits per char)
        std::string decoded_token = base64::decode(base64_token);
        
        this->token_map[decoded_token] = token_id;
    }
    
    DEBUG_LOG("Loaded ", this->token_map.size(), " tokens into tokenizer.");
}


std::vector<int> GreedyTokenizer::tokenize(std::string_view input) {
    std::vector<int> tokens;
    int pos = 0;
    while (pos < input.size()) {
        int longest_substr_len = 0;
        // Find the longest substring that is a key in token_map, or that hits end of input.
        for (int curr_len = 1; curr_len <= input.size() - pos; curr_len++) {
            auto substr = std::string(input.substr(pos, curr_len));
            if (this->token_map.contains(substr)) {
                longest_substr_len = curr_len;
            }
        }
        if (longest_substr_len == 0) {
            // TODO: handle unknown token
            DEBUG_LOG("Encountered unknown symbol within input: ", input);
        } 
        auto substr = std::string(input.substr(pos, longest_substr_len));
        tokens.push_back(this->token_map[substr]);
        pos += longest_substr_len;
    }
    return tokens;
}
// TODO: untokenize (for debugging and testing mostly)