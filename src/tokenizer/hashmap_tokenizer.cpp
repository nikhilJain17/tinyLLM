#include "hashmap_tokenizer.hpp"

HashMapTokenizer::HashMapTokenizer() {
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

std::vector<int> HashMapTokenizer::tokenize(std::string_view input) {
    /*
    Iterate over the whole string, keeping track of the lowest possible merge pair.
    Apply the merge, and continue until we reach a fixpoint.
    */
    
    // TODO: get rid of all these copies! Measure the perf difference when you do.
    std::vector<std::string> symbol_list;
    for (int i = 0; i < input.size(); i++) {
        symbol_list.push_back(std::string(1, input[i]));
    }
    std::cout << "Input:       " << input << "\n";
    std::cout << "Symbol list: ";
    for (auto s : symbol_list) {
        std::cout << s;
    }
    std::cout << "\n";

    std::vector<int> tokens;
    int lowest_merge_rank;
    int lowest_merge_index;
    do {
        lowest_merge_index = -1;
        lowest_merge_rank = 99999999;
        // Find the pair-wise merge with the lowest merge rank, if it exists
        for (int i = 0; i < symbol_list.size() - 1; i++) {
            std::string curr = symbol_list[i];
            std::string next = symbol_list[i+1];
            // TODO: get rid of all these copies! Measure the perf difference when you do.
            std::string curr_token = curr.append(next);
            if (this->token_map.contains(curr_token)) {
                if (this->token_map[curr_token] < lowest_merge_rank) {
                    lowest_merge_rank = this->token_map[curr_token];
                    lowest_merge_index = i;
                    std::cout << "Found a nice token: " << curr_token << ", of rank: " << lowest_merge_rank << "\n";
                }
            }
        }
        // Merge them
        if (lowest_merge_index >= 0) {
            std::string curr = symbol_list[lowest_merge_index];
            std::string next = symbol_list[lowest_merge_index+1];
            std::string curr_token = curr.append(next); 
            std::cout << "Merging: " << curr << " + " << next << " = " << curr_token << "\n";
            symbol_list[lowest_merge_index] = curr_token;
            symbol_list.erase(symbol_list.begin() + lowest_merge_index + 1);
        }
        
    } while (lowest_merge_index >= 0);
    for (auto symbol : symbol_list) {
        if (this->token_map.contains(symbol)) {
            tokens.push_back(this->token_map[symbol]);
        } else {
            DEBUG_LOG("Unknown symbol: ", symbol);
        }
    }
    return tokens;
}