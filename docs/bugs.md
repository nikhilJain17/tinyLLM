# Bugs.md


## Greedy Tokenizer bug

GreedyTokenizer is expected to fail this test case.
The string " (-0.6%)" in base64 encoding is ICjiiJIwLjYlKQ==
There are (more than) two possible tokenizations for " (-".
1. ICg=, which in UTF-8 is " ("
2. ICjiiA==, which in UTF-8 is " (�". In other words, it is the UTF-8 for 
" (" and the first 2 bytes of another 3 byte UTF-8 character! Hence the �.

This highlights two subtleties of BPE-based tokenizers. 
1. Token matching can be ambiguous, and we need to disambiguate not by longest prefix but by lowest rank.
2. Some tokens may contain partial characters, since BPE works at the byte level agnostic of encoding.

