#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

#define CHUNK 16777216

typedef struct {
    int num_tensors;
    
} Metadata;

typedef struct {
    std::vector<int> dim;
    std::vector<uint8_t> data;
} Tensor;

class WeightLoader
{
private:
    std::vector<uint8_t> buffer;
    const std::string filepath;
    bool parse_magic_number();
    int parse_gguf_version();
    // If the GPU has enough VRAM, load all tensors
    void load_fully_resident();
    // Otherwise, stream them to GPU
    bool stream_to_gpu();

public:
    WeightLoader(const std::string & filepath);
    Metadata get_metadata();
    Tensor fetch_tensor(std::string& tensor_name);
};

