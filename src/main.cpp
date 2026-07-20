#include "chrono"
#include "tokenizer/base64.hpp"
#include "tokenizer/hashmap_tokenizer.hpp"
#include "weightloader/weightloader.hpp"
#include <iostream>


int main() {
    DEBUG_LOG("Hello world");
    WeightLoader wl("/Users/njain/experiment/tinyLLM/resources/Llama-3.2-1B-Instruct-F16.gguf", Mode::ChunkedFullyResident);
    wl.write_token_vocab_to_file("/Users/njain/experiment/tinyLLM/resources/llama_tokens.txt");
    wl.write_token_merges_to_file("/Users/njain/experiment/tinyLLM/resources/llama_merges.txt");
}