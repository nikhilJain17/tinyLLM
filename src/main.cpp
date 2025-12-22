#include <iostream>
#include "tokenizer/greedy_tokenizer.hpp"
#include "tokenizer/base64.hpp"
int main() {
    GreedyTokenizer m;
    std::string f = "Hello world";
    for (int token : m.tokenize(f)) {
        std::cout << " " << token;
    }
    std::cout << std::endl;
    return 0;
}