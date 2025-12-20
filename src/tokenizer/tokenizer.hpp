#include <string>
#include <vector>

// Abstract class to be implemented
class Tokenizer {
public:
    virtual std::vector<int> tokenize(std::string&) = 0;
    virtual ~Tokenizer() = default;
};
