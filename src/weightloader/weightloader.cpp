#include "weightloader.hpp"

WeightLoader::WeightLoader(const std::string & filepath) : filepath(filepath) {
    this->buffer = std::vector<uint8_t>();
    // TODO: perf
    this->buffer.reserve(CHUNK);
}

bool WeightLoader::load_fully_resident() {
    
}