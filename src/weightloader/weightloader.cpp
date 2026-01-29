#include "weightloader.hpp"

WeightLoader::WeightLoader(const std::string &filepath) : filepath(filepath) {
  this->buffer = std::vector<uint8_t>();
  // TODO: perf
  this->buffer.reserve(CHUNK);
  this->load_fully_resident();
}

void WeightLoader::load_fully_resident() {
  DEBUG_LOG("Loading weights from ", this->filepath);
  int fd = open(this->filepath.c_str(), O_RDONLY);
  if (fd == -1) {
    DEBUG_LOG("Error opening file! ", this->filepath);
    exit(-1);
  }
  ssize_t bytes_read;
  ssize_t total_bytes_read = 0;
  while ((bytes_read = read(fd, this->buffer.data(), CHUNK)) > 0) {
    DEBUG_LOG("Read ", bytes_read, " bytes, with chunk size ", CHUNK,
              " bytes.");
    total_bytes_read += bytes_read;
  }
  DEBUG_LOG("Successfully loaded weights from ", this->filepath, " for ",
            total_bytes_read, " bytes.");
}