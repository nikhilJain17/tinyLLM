#include <iostream>
#include "tokenizer/hashmap_tokenizer.hpp"
#include "tokenizer/base64.hpp"
int main() {
    HashMapTokenizer m;
    std::string f = "490 (−0.6%)";
    for (int token : m.tokenize(f)) {
        std::cout << ", " << token;
    }
    std::cout << std::endl;
    return 0;
}