#include "tokenizer/greedy_tokenizer.hpp"
#include "tokenizer/hashmap_tokenizer.hpp"
#include <gtest/gtest.h>
#include <memory>

// Golden ids below are from HuggingFace tokenizers on
// unsloth/Llama-3.2-1B-Instruct (add_special_tokens=False).
class LlamaTokenizerTest : public ::testing::Test {
  protected:
	static std::unique_ptr<HashMapTokenizer> tok;
	static void SetUpTestSuite() {
		if (!tok) {
			tok = std::make_unique<HashMapTokenizer>(
				"resources/llama_tokens.txt", "resources/llama_merges.txt");
		}
	}
	static void TearDownTestSuite() { tok.reset(); }
};
std::unique_ptr<HashMapTokenizer> LlamaTokenizerTest::tok = nullptr;

// TODO: make these tests modular for swapping cl100k and o200k
TEST(TokenizerTest, StringOnly) {
	GreedyTokenizer tok;
	std::vector<int> actual = tok.tokenize("Hello world");
	std::vector<int> expected{9906, 1917};
	EXPECT_EQ(expected, actual);
}

TEST(TokenizerTest, NumbersOnly) {
	GreedyTokenizer tok;
	std::vector<int> actual = tok.tokenize(
		"12309393482090582798123093934820905827983123093934820905827"
		"98123093934820905827983");
	std::vector<int> expected{4513,	 25202, 24347, 18248, 22393, 24920, 25643,
							  9870,	 26164, 19746, 12652, 24824, 17267, 25009,
							  9870,	 26164, 19746, 12652, 24824, 17267, 19270,
							  15500, 18252, 21984, 18807, 23670, 26519, 18};

	EXPECT_EQ(expected, actual);
}

TEST(TokenizerTest, StringAndNumbersAndSpecialChars) {
	GreedyTokenizer tok;
	std::vector<int> actual = tok.tokenize(
		"The year is 2024! Email me@test.com or call +1-555-123-4567. !@#$^&*");
	std::vector<int> expected{791,	 1060,	374,  220, 2366,  19,  0,  8463,
							  757,	 48427, 916,  477, 1650,  489, 16, 12,
							  14148, 12,	4513, 12,  10961, 22,  13, 758,
							  31,	 49177, 61,	  5,   9};
	EXPECT_EQ(expected, actual);
}

TEST(TokenizerTest, AmbiguousCase) {
	/*
	GreedyTokenizer is expected to fail this test case.
	The string " (-0.6%)" in base64 encoding is ICjiiJIwLjYlKQ==
	There are (more than) two possible tokenizations for " (-".
	1. ICg=, which in UTF-8 is " ("
	2. ICjiiA==, which in UTF-8 is " (�". In other words, it is the UTF-8 for
	" (" and the first 2 bytes of another 3 byte UTF-8 character! Hence the �.

	This highlights two subtleties of BPE-based tokenizers.
	1. Token matching can be ambiguous, and we need to disambiguate not by
	longest prefix but by lowest rank.
	2. Some tokens may contain partial characters, since BPE works at the byte
	level agnostic of encoding.

	We include the naive encoder to highlight these implementation pitfalls.
	*/
	HashMapTokenizer tok;
	std::vector<int> actual = tok.tokenize("490 (−0.6%)");
	std::vector<int> expected{18518, 320, 34363, 15, 13, 21, 11587};
	EXPECT_EQ(expected.size(), actual.size());
	EXPECT_EQ(expected, actual);
}

TEST(TokenizerTest, LongTest1) {
	HashMapTokenizer tok;
	std::vector<int> actual = tok.tokenize(
		"Alexander the Great marched from Μακεδονία through Persia and Egypt, "
		"founding Alexandria; educated by Αριστοτέλης, he crossed the Hindu "
		"Kush, reached the Indus River, and is recorded as Ἀλέξανδρος in "
		"Greek, "
		"الإسكندر in Arabic, and Александр in Cyrillic—by 330 BCE the fall of "
		"Persepolis reshaped history; logistics note distances → ← ↑ ↓, "
		"temperatures −5 ℃ to 45 ℃, costs € and %, ratios 3:1, parentheses "
		"(routes, supply), brackets [sources], ellipses…, dashes — –, and "
		"symbols ± × ÷, while modern texts mix units, commas, semicolons; "
		"despite storms, deserts, and rivers, his empire spanned continents, "
		"climates, and cultures, ending in Babylon at age 32.");
	std::vector<int> expected{
		78850, 279,	  8681,	 59761, 505,   8008,  250,	 19481, 68437, 31243,
		86486, 28654, 34369, 55241, 19481, 1555,  21097, 689,	323,   15212,
		11,	   36330, 57233, 26,	33142, 555,	  8008,	 239,	39179, 30862,
		45028, 36924, 28654, 36924, 80531, 34586, 42524, 46742, 11,	   568,
		28129, 279,	  36142, 50275, 11,	   8813,  279,	 2314,	355,   11188,
		11,	   323,	  374,	 12715, 439,   87189, 120,	 230,	34586, 80531,
		138,   122,	  19481, 34369, 86486, 39179, 28654, 46742, 304,   18341,
		11,	   17607, 93062, 20665, 32173, 12061, 13628, 11318, 304,   35217,
		11,	   323,	  57855, 3114,	79288, 7486,  63873, 304,	95805, 416,
		2345,  1729,  220,	 10568, 79677, 279,	  4498,	 315,	3700,  325,
		10097, 285,	  64793, 10395, 3925,  26,	  43257, 5296,	27650, 11651,
		48564, 47463, 78954, 11,	20472, 25173, 20,	 29753, 225,   311,
		220,   1774,  29753, 225,	11,	   7194,  13281, 323,	1034,  11,
		42338, 220,	  18,	 25,	16,	   11,	  75075, 320,	20380, 11,
		8312,  705,	  40029, 510,	40751, 1145,  26689, 3153,	288,   1981,
		11,	   88646, 2001,	 1389,	11,	   323,	  18210, 20903, 25800, 1717,
		115,   11,	  1418,	 6617,	22755, 6651,  8316,	 11,	77702, 11,
		5347,  27561, 2439,	 26,	8994,  44583, 11,	 951,	15916, 11,
		323,   36617, 11,	 813,	32447, 9575,  19212, 66959, 11,	   92399,
		11,	   323,	  27833, 11,	13696, 304,	  64678, 520,	4325,  220,
		843,   13};
	EXPECT_EQ(expected.size(), actual.size());
	EXPECT_EQ(expected, actual);
}

TEST_F(LlamaTokenizerTest, HelloWorld) {
	std::vector<int> expected{9906, 1917};
	EXPECT_EQ(expected, tok->tokenize("Hello world"));
}

TEST_F(LlamaTokenizerTest, SimpleSentence) {
	std::vector<int> expected{791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679};
	EXPECT_EQ(expected, tok->tokenize("The quick brown fox jumps over the lazy dog"));
}

TEST_F(LlamaTokenizerTest, Prose) {
	std::vector<int> expected{8100, 15203, 14297, 3235, 279, 11594, 31284, 520, 39493};
	EXPECT_EQ(expected, tok->tokenize("She walked slowly along the quiet shore at dawn"));
}

TEST_F(LlamaTokenizerTest, Contraction) {
	std::vector<int> expected{40, 649, 956, 4510, 433, 596, 2736, 2884};
	EXPECT_EQ(expected, tok->tokenize("I can't believe it's already done"));
}

TEST_F(LlamaTokenizerTest, Multilingual) {
	std::vector<int> expected{78850, 59761, 1555, 21097, 689, 323, 15212, 11, 36330, 57233};
	EXPECT_EQ(expected,
			  tok->tokenize("Alexander marched through Persia and Egypt, founding Alexandria"));
}

TEST_F(LlamaTokenizerTest, PunctuationAndSmallNumbers) {
	std::vector<int> expected{880,	 716, 2859, 482, 20, 311, 220, 1774, 11, 42338,
							  220,	 18,  25,	16,	 11, 88646, 1198, 323, 18210};
	EXPECT_EQ(expected,
			  tok->tokenize("temperatures -5 to 45, ratios 3:1, dashes -- and symbols"));
}

// DISABLED until the pre-tokenization regex is implemented.
//
// The real Llama 3 tokenizer runs a regex (the cl100k/GPT-4 pattern) over the
// text BEFORE BPE, splitting it into chunks; merges are then only allowed
// within a chunk, never across chunk boundaries. Our tokenize() skips that step
// and feeds the whole string to BPE, so merges that the regex would have
// forbidden still fire.
//
// The gap shows up on:
//   - Multi-digit numbers: the pattern `\p{N}{1,3}` caps digit runs at 3, so
//     "2024" -> "202" + "4" (2366, 19). Without it, raw BPE merges the four
//     digits differently (508, 1187). Same for "14159" and the "4567" below.
//   - Punctuation runs butting against digits (the phone number).
//
// The expected ids here are the correct HF/Llama-3.2 golden values; these tests
// flip to enabled once the pre-tokenizer lands. See Meta's models/llama3/
// tokenizer.py (pat_str) and OpenAI tiktoken's cl100k_base pattern.
TEST_F(LlamaTokenizerTest, DISABLED_MultiDigitNumbers) {
	std::vector<int> expected{791, 1060, 374, 220, 2366, 19, 323,
							  9115, 374, 220, 18, 13, 9335, 2946};
	EXPECT_EQ(expected, tok->tokenize("The year is 2024 and pi is 3.14159"));
}

TEST_F(LlamaTokenizerTest, DISABLED_SpecialChars) {
	std::vector<int> expected{2386, 757, 48427, 916, 477, 1650, 489, 16, 12, 14148,
							  12, 4513, 12, 10961, 22, 758, 31, 49177, 46999, 5, 9};
	EXPECT_EQ(expected, tok->tokenize("email me@test.com or call +1-555-123-4567 !@#$%^&*"));
}
