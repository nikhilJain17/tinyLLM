# Performance.md
  
## General C++ Notes
- Use `std::unordered_map` instead of `std::map` because the former is a hashmap but the latter maintains keys in sorted order.  

- Always use `#pragma once` in headers to only define headers once.  

- Copying `std::strings` is expensive. Instead of using a reference type like `std::string&`, you can use `std::string_view`.  
  
## Tokenizer
  
### General Tokenizer Notes
     
### HashMapTokenizer
Unoptimized HashMapTokenizer Performance on Alexander test case:  
Loading tokens: 12ms  
Tokenizing: 4ms  
  
