#pragma once

#include <fstream>
#include <iostream>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "base64.hpp"
#include "tokenizer.hpp"

struct PairHash {
	size_t operator()(const std::pair<std::string, std::string>& p) const {
		return std::hash<std::string>{}(p.first) ^
			   (std::hash<std::string>{}(p.second) << 1);
	}
};

class HashMapTokenizer : public Tokenizer {
  public:
	HashMapTokenizer();
	HashMapTokenizer(std::string_view token_filepath, std::string_view token_merge_filepath);
	std::vector<int> tokenize(std::string_view input);

  private:
    // vocabulary of tokens
	std::unordered_map<std::string, int> token_map;
	// merge pair --> priority
	std::unordered_map<std::pair<std::string, std::string>, int, PairHash> token_merge_map;
	std::string token_filepath = "resources/cl100k_base.tiktoken";
	std::string token_merge_filepath;

	void load_vocab();
	void load_merges();
};
