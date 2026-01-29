#include "tokenizer/hashmap_tokenizer.hpp"

#include "chrono"
#include "iostream"

int main() {
  std::chrono::high_resolution_clock clock;
  std::string str =
      "Alexander the Great marched from Μακεδονία through Persia and Egypt, "
      "founding Alexandria; educated by Αριστοτέλης, he crossed the Hindu "
      "Kush, reached the Indus River, and is recorded as Ἀλέξανδρος in Greek, "
      "الإسكندر in Arabic, and Александр in Cyrillic—by 330 BCE the fall of "
      "Persepolis reshaped history; logistics note distances → ← ↑ ↓, "
      "temperatures −5 ℃ to 45 ℃, costs € and %, ratios 3:1, parentheses "
      "(routes, supply), brackets [sources], ellipses…, dashes — –, and "
      "symbols ± × ÷, while modern texts mix units, commas, semicolons; "
      "despite storms, deserts, and rivers, his empire spanned continents, "
      "climates, and cultures, ending in Babylon at age 32.";
  auto start = clock.now();
  HashMapTokenizer m;
  auto loaded_tokens = clock.now();
  std::vector tokens = m.tokenize(str);
  auto tokenized = clock.now();
  std::cout << "\n_________________\nPerformance\n";
  std::cout << "Loading tokens: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   loaded_tokens - start)
            << "\n";
  std::cout << "Tokenizing: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   tokenized - loaded_tokens)
            << "\n";
  return 0;
}