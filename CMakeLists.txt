cmake_minimum_required(VERSION 3.16)

project(tinyLLM LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Include all headers in src/ and subdirectories
include_directories(${PROJECT_SOURCE_DIR}/src)


# Recursively collect all cpp files in src/
file(GLOB_RECURSE SOURCES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# Create executable
add_executable(tinyLLM ${SOURCES})


# Forcefully include debug.h everywhere
target_precompile_headers(tinyLLM PRIVATE src/debug.h)


# Run target
# cmake --build build --target run
add_custom_target(run
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tinyLLM
    DEPENDS tinyLLM
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running tinyLLM..."
    USES_TERMINAL
)

# Debug logging
# cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
target_compile_definitions(tinyLLM PRIVATE
    $<$<CONFIG:Debug>:TINYLLM_DEBUG>
)