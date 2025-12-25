#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string_view>

#include "tokenizer.hpp"
#include "base64.hpp"

/**
 * This is a naive tokenizer which greedily does a substring match and hashmap lookup to match tokens.
 * This is the baseline to measure performance improvements.
 * It also highlights correctness pitfalls when tokenizing a byte-pair encoding based vocabulary.
    1. Token matching can be ambiguous, and we need to disambiguate not by longest prefix but by lowest rank.
    2. Some tokens may contain partial characters, since BPE works at the byte level agnostic of encoding.
 */
class GreedyTokenizer : public Tokenizer {
public:    
    GreedyTokenizer();
    GreedyTokenizer(std::string_view token_filepath);
    std::vector<int> tokenize(std::string_view input);
    
private:
    // TODO: Compare performance of a trie or memory-mapped file
    std::unordered_map<std::string, int> token_map;
    // TODO: (experiment) try other token vocabularies of different sizes and measure token/s
    std::string token_filepath = "resources/cl100k_base.tiktoken";
};
