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
    int start = 0;
    int end = 0;
    std::string substr = "";

    // load in token table as an enormous map
    std::ifstream ifs;
    ifs.open ("resources/cl100k_base.tiktoken", std::ifstream::in);
    char c;
    while (ifs.good()) {
        std::cout << c;
        c = ifs.get();
    }

  ifs.close();

    for (int i = 0; i < input.size(); i++) {
        // if (substr !=)
    }
    return tokens;
}
