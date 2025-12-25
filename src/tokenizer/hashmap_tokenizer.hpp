#include <fstream>
#include <iostream>
#include <unordered_map>
#include <string_view>

#include "tokenizer.hpp"
#include "base64.hpp"
class HashMapTokenizer: public Tokenizer {
public:    
    HashMapTokenizer();
    HashMapTokenizer(std::string_view token_filepath);
    std::vector<int> tokenize(std::string_view input);
    
private:
    std::unordered_map<std::string, int> token_map;
    std::string token_filepath = "resources/cl100k_base.tiktoken";
};
