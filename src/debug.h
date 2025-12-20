#pragma once

#ifdef TINYLLM_DEBUG
#include <iostream>
#include <cstring>
    #define DEBUG_LOG(msg) \
        do { \
            const char* file = strrchr(__FILE__, '/'); \
            if (file) file++; else file = __FILE__; \
            std::cerr << "[tinyLLM] " << file << ":" << __LINE__ \
                    << " " << msg << '\n'; \
        } while (0)
#else
    #define DEBUG_LOG(msg) do {} while (0)
#endif
