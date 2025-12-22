cmake_minimum_required(VERSION 3.16)
project(tinyLLM LANGUAGES CXX)

# Use C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# Include all headers in src/ and subdirectories
include_directories(${PROJECT_SOURCE_DIR}/src)

# Import GTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Recursively collect all cpp files in src/
file(GLOB_RECURSE SOURCES
    "${PROJECT_SOURCE_DIR}/src/*.cpp"
)

# Create executable
add_executable(tinyLLM ${SOURCES})

# Forcefully include debug.h everywhere
target_precompile_headers(tinyLLM PRIVATE src/debug.h)

# Run target
add_custom_target(run
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/tinyLLM
    DEPENDS tinyLLM
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Running tinyLLM..."
    USES_TERMINAL
)

# Debug build, enable logging 
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(tinyLLM PRIVATE TINYLLM_DEBUG)
endif()

# Function to create a test executable with source files, GTest, and precompiled headers
function(add_tinyllm_test test_name)
    set(TEST_SRC "${PROJECT_SOURCE_DIR}/test/${test_name}.cpp")
    if(NOT EXISTS ${TEST_SRC})
        message(FATAL_ERROR "Test source file ${TEST_SRC} does not exist")
    endif()
    
    # Collect all implementation sources BUT exclude main.cpp
    file(GLOB_RECURSE IMPLEMENTATION_SOURCES
        CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/src/*.cpp"
    )
    list(FILTER IMPLEMENTATION_SOURCES EXCLUDE REGEX ".*/main\\.cpp$")
    
    add_executable(${test_name}
        ${TEST_SRC}
        ${IMPLEMENTATION_SOURCES}
    )
    
    target_include_directories(${test_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/src
    )
    
    # Precompiled headers for test
    target_precompile_headers(${test_name} PRIVATE src/debug.h)
    
    target_link_libraries(${test_name} PRIVATE
        GTest::gtest_main
    )
    
    # Add to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Enable testing
enable_testing()

# For testing, copy the token file to the build directory.
configure_file(
    resources/cl100k_base.tiktoken
    ${CMAKE_BINARY_DIR}/resources/cl100k_base.tiktoken
    COPYONLY
)

# Add tests
add_tinyllm_test(tokenizer_test)

# A single custom target to rebuild everything and run tests
add_custom_target(run_tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --clean-first
    COMMAND ${CMAKE_CTEST_COMMAND} --test-dir ${CMAKE_BINARY_DIR} --verbose
    DEPENDS tokenizer_test tinyLLM
    COMMENT "Rebuilding all and running all tests..."
    USES_TERMINAL
)