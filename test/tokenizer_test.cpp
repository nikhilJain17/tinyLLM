#include <gtest/gtest.h>
#include "tokenizer/greedy_tokenizer.hpp"

// TODO: make these tests modular for swapping cl100k and o200k
TEST(TokenizerTest, StringOnly) {
    GreedyTokenizer tok;
    std::vector<int> actual = tok.tokenize("Hello world");
    std::vector<int> expected {9906, 1917};
    EXPECT_EQ(expected, actual);
}


TEST(TokenizerTest, NumbersOnly) {
    GreedyTokenizer tok;
    std::vector<int> actual = tok.tokenize("1230939348209058279812309393482090582798312309393482090582798123093934820905827983");
    std::vector<int> expected {4513, 25202, 24347, 18248, 22393, 24920, 25643, 
        9870, 26164, 19746, 12652, 24824, 17267, 25009, 9870, 26164, 19746, 12652, 
        24824, 17267, 19270, 15500, 18252, 21984, 18807, 23670, 26519, 18};
    
    EXPECT_EQ(expected, actual);
}


TEST(TokenizerTest, StringAndNumbersAndSpecialChars) {
    GreedyTokenizer tok;
    std::vector<int> actual = tok.tokenize("The year is 2024! Email me@test.com or call +1-555-123-4567. !@#$^&*");
    std::vector<int> expected {791, 1060, 374, 220, 2366, 19, 0, 8463, 757, 48427, 916, 477, 1650, 489, 16, 12, 
        14148, 12, 4513, 12, 10961, 22, 13, 758, 31, 49177, 61, 5, 9};
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
    1. Token matching can be ambiguous, and we need to disambiguate not by longest prefix but by lowest rank.
    2. Some tokens may contain partial characters, since BPE works at the byte level agnostic of encoding.

    We include the naive encoder to highlight these implementation pitfalls.
    */
    GreedyTokenizer tok;
    std::vector<int> actual = tok.tokenize("490 (−0.6%)");
    std::vector<int> expected {18518, 320, 34363, 15, 13, 21, 11587, 220};
    EXPECT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        std::cout << i << " is " << (expected[i] == actual[i] ? "OK, " : " WRONG, ") 
        << tok.token_id_to_str_map[expected[i]] << " -> " << expected[i]
        << " and got " 
        << tok.token_id_to_str_map[actual[i]] << " -> " << actual[i] << "\n";
        if (expected[i] != actual[i]) {
            std::cout << "Expected " << tok.token_id_to_str_map[expected[i]] << " -> " << expected[i] 
                << " but got " << tok.token_id_to_str_map[actual[i]] << " -> " << actual[i] << "\n";
        }
    }
    EXPECT_EQ(expected, actual);
}
