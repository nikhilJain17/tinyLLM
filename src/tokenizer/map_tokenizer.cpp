#include "map_tokenizer.hpp"

MapTokenizer::MapTokenizer() {
    // Load in token table into memory
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
std::vector<int> MapTokenizer::tokenize(std::string& input) {
    std::vector<int> tokens;
    int substr_size = 0;
    std::string_view substr;

    for (int i = 0; i < input.size(); i++) {
        substr_size++;
        substr = std::string_view(input.c_str(), substr_size);
        /*
        TODO: convert input to base64 encoded string.
        */
        if (!this->token_map.contains(substr)) {

        }
    }
    return tokens;
}
