# tinyLLM
Yet another LLM inference engine

## Building
mkdir -p build
cmake -S . -B build
cmake --build build

## Run directly
./build/bin/tinyLLM

## Or using the CMake run target
cmake --build build --target run
