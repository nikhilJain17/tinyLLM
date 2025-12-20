#include <fstream>
#include <iostream>
#include <map>

#include "tokenizer.hpp"

class MapTokenizer : public Tokenizer {
public:    
    MapTokenizer();
    MapTokenizer(std::string& token_filepath);
    std::vector<int> tokenize(std::string& input);
private:
    std::map<std::string, int> token_map;
    std::string token_filepath = "resources/cl100k_base.tiktoken";
};
