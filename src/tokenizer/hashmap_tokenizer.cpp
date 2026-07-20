#include "hashmap_tokenizer.hpp"

#include <array>

namespace {

const std::array<std::string, 256>& byte_to_unicode() {
	static const std::array<std::string, 256> table = [] {
		std::array<int, 256> cp{};
		std::array<bool, 256> direct{};
		auto keep = [&](int lo, int hi) {
			for (int b = lo; b <= hi; b++) {
				cp[b] = b;
				direct[b] = true;
			}
		};
		keep('!', '~');
		keep(0xA1, 0xAC);
		keep(0xAE, 0xFF);
		int n = 0;
		for (int b = 0; b < 256; b++) {
			if (!direct[b]) {
				cp[b] = 256 + n++;
			}
		}
		std::array<std::string, 256> t;
		for (int b = 0; b < 256; b++) {
			int c = cp[b];
			if (c < 0x80) {
				t[b] = std::string(1, static_cast<char>(c));
			} else {
				t[b] = {static_cast<char>(0xC0 | (c >> 6)),
						static_cast<char>(0x80 | (c & 0x3F))};
			}
		}
		return t;
	}();
	return table;
}

} // namespace

HashMapTokenizer::HashMapTokenizer() {
	load_vocab();
}

HashMapTokenizer::HashMapTokenizer(
	std::string_view token_filepath,
	std::string_view token_merge_filepath)
	: token_filepath(token_filepath),
	  token_merge_filepath(token_merge_filepath) {
	load_vocab();
	load_merges();
}

void HashMapTokenizer::load_vocab() {
	// TODO: compare with memory mapped tokenizer
	std::ifstream ifs(this->token_filepath);
	if (!ifs) {
		throw std::runtime_error("Failed to open token file: " +
								 this->token_filepath);
	}

	std::string line;
	while (std::getline(ifs, line)) {
		// "<base64_token> <id>"
		size_t space_pos = line.rfind(' ');
		if (space_pos == std::string::npos) {
			continue; // Skip malformed lines
		}
		std::string decoded_token = base64::decode(line.substr(0, space_pos));
		int token_id = std::stoi(line.substr(space_pos + 1));
		this->token_map[decoded_token] = token_id;
	}

	DEBUG_LOG("Loaded ", this->token_map.size(), " tokens into tokenizer.");
}

void HashMapTokenizer::load_merges() {
	std::ifstream ifs(this->token_merge_filepath);
	if (!ifs) {
		throw std::runtime_error("Failed to open merges file: " +
								 this->token_merge_filepath);
	}

	std::string line;
	int rank = 0;
	while (std::getline(ifs, line)) {
		// "<left> <right>", rank = line order
		size_t space_pos = line.find(' ');
		if (space_pos == std::string::npos) {
			continue;
		}
		std::string left = line.substr(0, space_pos);
		std::string right = line.substr(space_pos + 1);
		this->token_merge_map[{left, right}] = rank++;
	}

	DEBUG_LOG("Loaded ", this->token_merge_map.size(), " merges into tokenizer.");
}

std::vector<int> HashMapTokenizer::tokenize(std::string_view input) {
	bool use_merges = !this->token_merge_map.empty();

	std::vector<std::string> symbol_list;
	if (use_merges) {
		const auto& b2u = byte_to_unicode();
		for (unsigned char c : input) {
			symbol_list.push_back(b2u[c]);
		}
	} else {
		for (unsigned char c : input) {
			symbol_list.push_back(std::string(1, c));
		}
	}

	std::vector<int> tokens;
	int lowest_merge_rank;
	int lowest_merge_index;
	do {
		lowest_merge_index = -1;
		lowest_merge_rank = 99999999;
		for (size_t i = 0; i + 1 < symbol_list.size(); i++) {
			const std::string& curr = symbol_list[i];
			const std::string& next = symbol_list[i + 1];
			int rank;
			if (use_merges) {
				auto it = this->token_merge_map.find({curr, next});
				if (it == this->token_merge_map.end()) {
					continue;
				}
				rank = it->second;
			} else {
				auto it = this->token_map.find(curr + next);
				if (it == this->token_map.end()) {
					continue;
				}
				rank = it->second;
			}
			if (rank < lowest_merge_rank) {
				lowest_merge_rank = rank;
				lowest_merge_index = i;
			}
		}
		if (lowest_merge_index >= 0) {
			symbol_list[lowest_merge_index] += symbol_list[lowest_merge_index + 1];
			symbol_list.erase(symbol_list.begin() + lowest_merge_index + 1);
		}
	} while (lowest_merge_index >= 0);

	for (const auto& symbol : symbol_list) {
		auto it = this->token_map.find(symbol);
		if (it != this->token_map.end()) {
			tokens.push_back(it->second);
		} else {
			DEBUG_LOG("Unknown symbol: ", symbol);
		}
	}
	return tokens;
}