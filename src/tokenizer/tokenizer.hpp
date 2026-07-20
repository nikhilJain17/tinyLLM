#pragma once

#include <string>
#include <vector>

// Only accept tiktoken-style BPE vocabularies for now.
class Tokenizer {
  public:
	virtual std::vector<int> tokenize(std::string_view) = 0;
	virtual ~Tokenizer() = default;
};
