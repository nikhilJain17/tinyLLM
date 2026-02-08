#include "chrono"
#include "tokenizer/base64.hpp"
#include "tokenizer/hashmap_tokenizer.hpp"
#include "weightloader/weightloader.hpp"
#include <iostream>

int main() {
	WeightLoader w(
		"resources/ggml-org_models_tinyllamas_stories15M-q4_0.gguf");
}