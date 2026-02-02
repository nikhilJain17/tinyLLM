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

uint64_t WeightLoader::parse_tensor_count() {
	return this->peek_u64_little_endian(size_t(8));
}

uint64_t WeightLoader::parse_metadata_kv_count() {
	return this->peek_u64_little_endian(16);
}

// STOPSHIP: parse metadata kv pairs
std::unordered_map<std::string, MetadataValue> WeightLoader::parse_metadata_kv_pairs() {
	int metadata_kv_count = this->parse_metadata_kv_count();
	std::unordered_map<std::string, MetadataValue> metadata;
	size_t cursor = 24;
	for (int i = 0; i < metadata_kv_count; i++) {
		// First 8 bytes is key_len
		// Next key_len bytes is the key
		// Next 4 bytes is value type
		// Then proceed from there
		uint64_t key_len = consume_u64_little_endian(cursor);
		std::string key = this->consume_str(cursor, key_len);
		std::cout << "key: " << key <<  ", cursor: " << cursor << "\n";

		MetadataType type = static_cast<MetadataType>(this->consume_u32_little_endian(cursor));
		// std::cout << "metadata type: " << (type) << "\n";
		switch (type) {
			case MetadataType::str: {
				uint64_t str_len = consume_u64_little_endian(cursor);
				std::string value = this->consume_str(cursor, str_len);
				std::cout << "> value str: " << value << "\n";
				metadata.try_emplace(key, MetadataValue{type, value});
				break;
			}
			case MetadataType::u32: {
				uint32_t value = this->consume_u32_little_endian(cursor);
				std::cout << "> value u32: " << value << "\n";
				metadata.try_emplace(key, MetadataValue{type, value});
				break;
			}
			case MetadataType::u64: {
				uint64_t value = this->consume_u32_little_endian(cursor);
				std::cout << "> value u64: " << value << "\n";
				metadata.try_emplace(key, MetadataValue{type, value});
				break;
			}
			case MetadataType::array: {
				std::vector<MetadataValue> array_vals = this->consume_array(cursor);	

				std::cout << "> value array_vals:\n";
				for (const auto& val : array_vals) {
					std::cout << "\t";
					std::visit([](const auto& v) {
						using T = std::decay_t<decltype(v)>;
						if constexpr (std::is_same_v<T, std::vector<MetadataValue>>) {
							std::cout << "[array size=" << v.size() << "]";
						} else if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>) {
							std::cout << static_cast<int>(v); // numeric, not char
						} else {
							std::cout << v;
						}
					}, val.payload);
					std::cout << "\n";
				}

				metadata.try_emplace(key, MetadataValue{type, array_vals});
				break;
			}
			default: {
				throw std::invalid_argument("got invalid metadata value type: " + std::to_string(type));
			}
		}
	

	}
	return metadata;
}

uint64_t WeightLoader::peek_u64_little_endian(size_t index) {
	uint64_t result = 0;
	for (int i = 0; i < 8; i++) {
		if (index + i > this->buf_size) {
			throw std::invalid_argument("trying to read beyond buffer size!");
		}
		result += this->buffer[index + i] << (8 * i);
	}
	return result;
}

uint32_t WeightLoader::peek_u32_little_endian(size_t index) {
	uint64_t result = 0;
	for (int i = 0; i < 4; i++) {
		if (index + i > this->buf_size) {
			throw std::invalid_argument("trying to read beyond buffer size!");
		}
		result += this->buffer[index + i] << (8 * i);
	}
	return result;
}

uint64_t WeightLoader::consume_u64_little_endian(size_t& cursor) {
	uint64_t result = peek_u64_little_endian(cursor);
	cursor += 8;
	return result;
}

uint32_t WeightLoader::consume_u32_little_endian(size_t& cursor) {
	uint32_t result = peek_u32_little_endian(cursor);
	cursor += 4;
	return result;
}

std::string WeightLoader::consume_str(size_t& cursor, size_t size) {
    if (cursor + size > buf_size) {
        throw std::invalid_argument("trying to read beyond buffer size!");
    }
    const char* p = reinterpret_cast<const char*>(buffer.get() + cursor);
    std::string result(p, p + size);   
    cursor += size;
    return result;
}

std::vector<MetadataValue> WeightLoader::consume_array(size_t& cursor) {
	// 32 bits: type of array
	// 64 bits: array length
	// remaining bits: data
	MetadataType arr_type = static_cast<MetadataType>(this->consume_u32_little_endian(cursor));
	uint64_t arr_len = this->consume_u64_little_endian(cursor);
	std::vector<MetadataValue> arr_vals;
	std::cout << "Parsing array of type: " << arr_type << "\n";
	switch (arr_type) {
		case MetadataType::str: {
			for (int i = 0; i < arr_len; i++) {
				uint64_t str_len = this->consume_u64_little_endian(cursor);
				std::string str = this->consume_str(cursor, str_len);
				arr_vals.push_back(MetadataValue{arr_type, str});
			}
			break;
		}
		case MetadataType::u32: {
			for (int i = 0; i < arr_len; i++) {
				uint32_t val = this->consume_u32_little_endian(cursor);
				arr_vals.push_back(MetadataValue{arr_type, val});
			}
			break;
		}
		case MetadataType::u64: {
			for (int i = 0; i < arr_len; i++) {
				uint64_t val = this->consume_u64_little_endian(cursor);
				arr_vals.push_back(MetadataValue{arr_type, val});
			}
			break;
		}
		default: {
			throw std::invalid_argument("invalid array type: " + std::to_string(arr_type));
			break;
		}
	}
	return arr_vals;
}
