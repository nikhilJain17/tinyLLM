#include "weightloader.hpp"
#include <iomanip>
WeightLoader::WeightLoader(const std::string &filepath) : filepath(filepath) {
	this->buffer = std::vector<uint8_t>();
	// TODO: perf
	this->buffer.reserve(CHUNK);
	this->load_fully_resident();
	DEBUG_LOG("Magic number is ",
			  this->parse_magic_number() == 1 ? "VALID" : "NOT VALID");
}

void WeightLoader::load_fully_resident() {
	DEBUG_LOG("Loading weights from ", this->filepath);
	int fd = open(this->filepath.c_str(), O_RDONLY);
	if (fd == -1) {
		DEBUG_LOG("Error opening file! ", this->filepath);
		exit(-1);
	}
	ssize_t bytes_read = 0;
	ssize_t total_bytes_read = 0;
	while ((bytes_read = read(fd, this->buffer.data() + bytes_read, CHUNK)) >
		   0) {
		DEBUG_LOG("Read ", bytes_read, " bytes, with chunk size ", CHUNK,
				  " bytes.");
		total_bytes_read += bytes_read;
	}
	DEBUG_LOG("Successfully loaded weights from ", this->filepath, " for ",
			  total_bytes_read, " bytes.");
}

bool WeightLoader::parse_magic_number() {
	return this->buffer[0] == 0x47 && this->buffer[1] == 0x47 &&
		   this->buffer[2] == 0x55 && this->buffer[3] == 0x46;
}
