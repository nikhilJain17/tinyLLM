#include "chrono"
#include "tokenizer/base64.hpp"
#include "tokenizer/hashmap_tokenizer.hpp"
#include "weightloader/weightloader.hpp"
#include <iostream>

int main() {
  WeightLoader w(
      "resources/ggml-org_gemma-3-1b-it-GGUF_gemma-3-1b-it-Q4_K_M.gguf");
}