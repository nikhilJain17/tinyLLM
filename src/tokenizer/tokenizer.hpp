#pragma once

#include <string>
#include <vector>

// Abstract class to be implemented
class Tokenizer {
public:
    virtual std::vector<int> tokenize(std::string_view) = 0;
    virtual ~Tokenizer() = default;  
};
