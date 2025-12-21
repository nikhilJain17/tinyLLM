# tinyLLM
Yet another LLM inference engine

## Build
mkdir -p build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

## Build debug
mkdir -p build
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug

## Run binary 
./build/bin/tinyLLM

### Or run via the CMake run target
cmake --build build --target run

### For running debug
cmake --build build-debug --target run

