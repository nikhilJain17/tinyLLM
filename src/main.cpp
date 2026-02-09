#include "chrono"
#include "tokenizer/base64.hpp"
#include "tokenizer/hashmap_tokenizer.hpp"
#include "weightloader/weightloader.hpp"
#include <iostream>


void sequential_test() {
	WeightLoader w(
		"resources/ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-llama-3.1-8b-instruct-q4_0.gguf");
		std::cout << "Done with 1\n";
	std::unordered_map<std::string, TensorInfo> tensor_index = w.get_tensor_index();
		std::cout << "Done with 2\n";
	for (auto &kv : tensor_index) {
		std::cout << "Fetching: " << kv.first << "\n";
		w.fetch_tensor(kv.first);
	}

}


int main() {
	sequential_test();	
}