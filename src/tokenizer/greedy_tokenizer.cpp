#include "greedy_tokenizer.hpp"

GreedyTokenizer::GreedyTokenizer() {
    // Load in token table into memory
    // TODO: thesis section: compare naive loading with memory mapping
    std::ifstream ifs;
    ifs.open(this->token_filepath, std::ifstream::in);
    std::string token;
    int token_id;
    while (ifs >> token >> token_id) {
        this->token_map[token] = token_id;
    }
    ifs.close();
    DEBUG_LOG("Loaded " << this->token_map.size() << " tokens into tokenizer.");
}

// tokenize :: string -> vector<int>
// take in a string and convert it to a list of tokens
// TODO: make sure to handle eof or end of input etc
std::vector<int> GreedyTokenizer::tokenize(std::string_view input) {
    std::vector<int> tokens;
    int substr_pos = 0;
    int substr_size = 0;
    std::string_view substr;

    for (int i = 0; i < input.size(); i++) {
        substr_size++;
        substr = input.substr(substr_pos, substr_size);
        std::string encoded_substr = base64::encode(substr);
        if (!this->token_map.contains(encoded_substr)) {
            auto sv = input.substr(substr_pos, substr_size - 1);
            std::string longest_matching_substr = std::string(sv);
            int token = this->token_map[longest_matching_substr];
            DEBUG_LOG(
                "Longest matched substring: " << longest_matching_substr 
                << ", encoding: " << base64::encode(substr) 
                << ", token: " << token
            );
            tokens.push_back(token);
            substr_size = 0;
            substr_pos = i;
        }
    }
    return tokens;
}
