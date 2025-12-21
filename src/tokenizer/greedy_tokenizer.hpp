#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string_view>

#include "tokenizer.hpp"
#include "base64.hpp"

/**
 * This is a naive tokenizer which greedily does a substring match and hashmap lookup to match tokens.
 * This is the baseline to measure performance improvements from byte-pair encoding tokenization.
 */
class GreedyTokenizer : public Tokenizer {
public:    
    GreedyTokenizer();
    GreedyTokenizer(std::string_view token_filepath);
    std::vector<int> tokenize(std::string_view input);
private:
    std::unordered_map<std::string, int> token_map;
    // TODO: thesis section: experiment with other token vocabularies and measure token/s
    std::string token_filepath = "resources/cl100k_base.tiktoken";
};
