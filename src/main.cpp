#include "chrono"
#include "tokenizer/base64.hpp"
#include "tokenizer/hashmap_tokenizer.hpp"
#include "weightloader/weightloader.hpp"
#include <iostream>

void sequential_test(std::string model_filepath, bool fully_resident) {
	std::chrono::high_resolution_clock clock;
	auto start = clock.now();
	WeightLoader w(model_filepath, fully_resident);
	std::unordered_map<std::string, TensorInfo> tensor_index =
		w.get_tensor_index();
	for (auto &kv : tensor_index) {
		w.fetch_tensor(kv.first);
	}
	auto end = clock.now();
	std::cout << "[   SEQUENTIAL TEST   ]\n"
			  << "\t- " << model_filepath << "\n"
			  << "\t- "
			  << std::chrono::duration_cast<std::chrono::milliseconds>(end -
																	   start)
			  << "\n";
}

int main() {
	sequential_test(
		"resources/"
		"ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-llama-3.1-8b-"
		"instruct-q4_0.gguf",
		// "resources/ggml-org_models_tinyllamas_stories15M-q4_0.gguf",
		true);
}