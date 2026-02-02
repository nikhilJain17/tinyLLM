#include "weightloader.hpp"
WeightLoader::WeightLoader(const std::string &filepath) : filepath(filepath) {
	this->load_fully_resident();
	DEBUG_LOG("Magic number is ",
			  this->parse_magic_number() == 1 ? "VALID" : "NOT VALID");
	DEBUG_LOG("GGUF version ", this->parse_gguf_version());
	DEBUG_LOG("Contains ", this->parse_tensor_count(), " tensors.");
	DEBUG_LOG("Contains ", this->parse_metadata_kv_count(),
			  " metadata kv pairs.");

	this->parse_metadata_kv_pairs();
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
	this->buf_size = st.st_size;
	this->buffer = std::make_unique<uint8_t[]>(this->buf_size);
	while ((bytes_read =
				read(fd, this->buffer.get() + total_bytes_read, st.st_size)) > 0) {
		DEBUG_LOG("Read ", bytes_read, "/", st.st_size, " bytes, or ",
				  (bytes_read / st.st_size) * 100, "% of the file.");
		total_bytes_read += bytes_read;
	}
	DEBUG_LOG("Successfully loaded weights from ", this->filepath, " totaling ",
			  total_bytes_read, " bytes.");
	std::cout << "Buffer.size: " << this->buf_size << "\n";
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
	return this->parse_u64_little_endian(8);
}

int WeightLoader::parse_metadata_kv_count() {
	return this->parse_u64_little_endian(16);
}

// STOPSHIP: parse metadata kv pairs
std::unordered_map<std::string, MetadataValue> WeightLoader::parse_metadata_kv_pairs() {
	int metadata_kv_count = this->parse_metadata_kv_count();
	std::unordered_map<std::string, MetadataValue> metadata;
	int cursor = 24;
	for (int i = 0; i < metadata_kv_count; i++) {
		// First 8 bytes is key_len
		// Next key_len bytes is the key
		// Next 4 bytes is value type
		// Then proceed from there
		uint64_t key_len = parse_u64_little_endian(cursor);
		std::cout << "key len: " << key_len << "\n";
		cursor += 8;
		std::string key = this->parse_str(cursor, key_len);
		cursor += key_len;
		std::cout << "key: " << key << "\n";
		break;  

	}
	return metadata;
}

uint64_t WeightLoader::parse_u64_little_endian(int index) {
	uint64_t result = 0;
	for (int i = 0; i < 4; i++) {
		result += this->buffer[index + i] << (8 * i);
	}
	return result;
}

std::string WeightLoader::parse_str(int index, int size) {
	std::string result;
	for (int i = index; i < index + size; i++) {
		// if (i > this->buffer.size()) {
		// 	std::cout << "wtf: " << i << " > " << this->buffer.size();
		// 	exit(1);
		// }
		result += this->buffer[i];
	}
	return result;
}