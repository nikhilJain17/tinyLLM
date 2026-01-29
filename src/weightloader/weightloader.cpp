#include "weightloader.hpp"
WeightLoader::WeightLoader(const std::string &filepath) : filepath(filepath) {
	this->buffer = std::vector<uint8_t>();
	this->load_fully_resident();
	DEBUG_LOG("Magic number is ",
			  this->parse_magic_number() == 1 ? "VALID" : "NOT VALID");
	DEBUG_LOG("GGUF version ", this->parse_gguf_version());
	DEBUG_LOG("Loaded ", this->parse_tensor_count(), " tensors.");
	DEBUG_LOG("Loaded ", this->parse_metadata_kv_count(),
			  " metadata kv pairs.");
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
	uint32_t gguf_version = 0;
	for (int i = 0; i < 4; i++) {
		gguf_version += this->buffer[4 + i] << (8 * i);
	}
	return gguf_version;
}

int WeightLoader::parse_tensor_count() {
	uint64_t tensor_count = 0;
	for (int i = 0; i < 8; i++) {
		tensor_count += this->buffer[8 + i] << (8 * i);
	}
	return tensor_count;
}

int WeightLoader::parse_metadata_kv_count() {
	uint64_t metadata_kv_count = 0;
	for (int i = 0; i < 8; i++) {
		metadata_kv_count += this->buffer[16 + i] << (8 * i);
	}
	return metadata_kv_count;
}

// STOPSHIP: parse metadata kv pairs
