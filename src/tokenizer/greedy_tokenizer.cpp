#include "greedy_tokenizer.hpp"

GreedyTokenizer::GreedyTokenizer() {
    // Load in token table into memory
    // TODO: (experiment) compare naive loading with memory mapping
    std::ifstream ifs;
    ifs.open(this->token_filepath, std::ifstream::in);
    std::string token;
    int token_id;
    while (ifs >> token >> token_id) {
        this->token_map[token] = token_id;
    }
    ifs.close();
    DEBUG_LOG("Loaded ", this->token_map.size(), " tokens into tokenizer.");
}

std::vector<int> GreedyTokenizer::tokenize(std::string_view input) {
    std::vector<int> tokens;
    int pos = 0;
    while (pos < input.size()) {
        int longest_substr_len = 0;
        // Find the longest substring that is a key in token_map, or that hits end of input.
        for (int curr_len = 1; curr_len < input.size(); curr_len++) {
            auto substr = std::string(input.substr(pos, curr_len));
            auto encoded = base64::encode(substr);
            if (this->token_map.contains(encoded)) {
                longest_substr_len = curr_len;
            }
        }
        // TODO: handle unknown token
        if (longest_substr_len == 0) {
            DEBUG_LOG("Encountered unknown symbol within input: ", input);
        }
        auto substr = std::string(input.substr(pos, longest_substr_len));
        auto encoded = base64::encode(substr);
        tokens.push_back(this->token_map[encoded]);
        pos += longest_substr_len;
    }
    return tokens;
}
// TODO: untokenize (for debugging and testing mostly)