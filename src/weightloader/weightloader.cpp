#include "weightloader.hpp"
WeightLoader::WeightLoader(const std::string &filepath) : filepath(filepath) {
	this->buffer = std::vector<uint8_t>();
	this->load_fully_resident();
	DEBUG_LOG("Magic number is ",
			  this->parse_magic_number() == 1 ? "VALID" : "NOT VALID");
	DEBUG_LOG("GGUF version is ", parse_gguf_version());
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
	struct stat st;
	stat(this->filepath.c_str(), &st);
	this->buffer.reserve(st.st_size);
	while ((bytes_read =
				read(fd, this->buffer.data() + bytes_read, st.st_size)) > 0) {
		DEBUG_LOG("Read ", bytes_read, "/", st.st_size, " bytes, or ",
				  (bytes_read / st.st_size) * 100, "% of the file.");
		total_bytes_read += bytes_read;
	}
	DEBUG_LOG("Successfully loaded weights from ", this->filepath, " totaling ",
			  total_bytes_read, " bytes.");
}

bool WeightLoader::parse_magic_number() {
	return this->buffer[0] == 0x47 && this->buffer[1] == 0x47 &&
		   this->buffer[2] == 0x55 && this->buffer[3] == 0x46;
}

int WeightLoader::parse_gguf_version() {
	return (static_cast<uint32_t>(buffer[4]) << 0) |
		   (static_cast<uint32_t>(buffer[5]) << 8) |
		   (static_cast<uint32_t>(buffer[6]) << 16) |
		   (static_cast<uint32_t>(buffer[7]) << 24);
}