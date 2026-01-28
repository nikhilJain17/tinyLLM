#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define CHUNK 16777216

class WeightLoader
{
private:
    std::vector<uint8_t> buffer;
    const std::string filepath;
    bool check_magic_number();
    int parse_gguf_version();


public:
    WeightLoader(const std::string & filepath);
    // If the GPU has enough VRAM, load all tensors
    bool load_fully_resident();
    // Otherwise, stream them to GPU
    bool stream_to_gpu();
};

