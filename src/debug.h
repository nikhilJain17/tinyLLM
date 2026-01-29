#pragma once
#ifdef TINYLLM_DEBUG
#include <cstring>
#include <iostream>

namespace detail {
template <typename... Args>
void debug_log_impl(const char *file, int line, Args &&...args) {
  std::cerr << "[tinyLLM] " << file << ":" << line << " ";
  (std::cerr << ... << std::forward<Args>(args)) << '\n';
}
} // namespace detail

#define DEBUG_LOG(...)                                                         \
  do {                                                                         \
    const char *file = strrchr(__FILE__, '/');                                 \
    if (file)                                                                  \
      file++;                                                                  \
    else                                                                       \
      file = __FILE__;                                                         \
    ::detail::debug_log_impl(file, __LINE__, __VA_ARGS__);                     \
  } while (0)
#else
#define DEBUG_LOG(...)                                                         \
  do {                                                                         \
  } while (0)
#endif