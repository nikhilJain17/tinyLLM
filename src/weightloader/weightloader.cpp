#include "weightloader.hpp"
WeightLoader::WeightLoader(const std::string &filepath) : filepath(filepath) {
	this->load_fully_resident();
	DEBUG_LOG("Magic number is ",
			  this->parse_magic_number() == 1 ? "VALID" : "NOT VALID");
	DEBUG_LOG("GGUF version ", this->parse_gguf_version());
	DEBUG_LOG("Contains ", this->parse_tensor_count(), " tensors.");
	DEBUG_LOG("Contains ", this->parse_metadata_kv_count(),
			  " metadata kv pairs.");

	size_t cursor = this->parse_metadata_kv_pairs();
	DEBUG_LOG("Successfully parsed ", this->metadata.size(), " metadata kv pairs.");
#ifdef TINYLLM_DEBUG
	this->dump_metadata();
#endif
	this->parse_tensor_info(cursor);
#ifdef TINYLLM_DEBUG
	this->dump_tensor_info();
#endif
	DEBUG_LOG("Successfully parsed ", this->tensor_index.size(), " tensor info structs.");
	std::optional<TensorView> t = this->fetch_tensor("blk.1.ffn_up.weight");
	if (t.has_value()) {
		this->dump_tensor_view(t.value());
	} else {
		std::cout << "Did not get tensor view\n";
	}
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

std::optional<TensorView> WeightLoader::fetch_tensor(std::string_view tensor_name) {
	std::string key(tensor_name);
	auto it = tensor_index.find(key);
	if (it == tensor_index.end()) {
		return std::nullopt;
	}
	const TensorInfo& info = it->second;
	TensorView t;
	t.name = info.name;
	t.data = this->buffer.get() + info.offset;
	size_t num_elements = 1; 
	for (int i = 0; i < info.n_dim; i++) {
		num_elements *= info.dim[i];
	}
	t.type = info.type;
	switch (t.type) {
		case TensorType::GGML_TYPE_F32:
			t.n_bytes = num_elements * 4;
			break;
		case TensorType::GGML_TYPE_F16:
			t.n_bytes = num_elements * 2;
			break;
		case TensorType::GGML_TYPE_BF16:
			t.n_bytes = num_elements * 2;
			break;
		case TensorType::GGML_TYPE_Q4_0:
			// Q4_0 blocks have one f32 for scale and 32 values of 4 bits each.
			// [ scale | v0 v1 ... v31 ]
			// There are 32 values in a block, and each block is 20 bytes.
			// 4 bytes for scale, and 4 bits * 32 values = 16 bytes for the values.
			t.n_bytes = std::ceil(num_elements / 32) * 20;
			break;
		case TensorType::GGML_TYPE_Q4_1:
			break;
		default:
			throw std::invalid_argument("unsupported tensor type");
	}
	t.dim = std::span<const uint64_t>(info.dim.data(), info.dim.size());
	return t;
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

size_t WeightLoader::parse_metadata_kv_pairs() {
	int metadata_kv_count = this->parse_metadata_kv_count();	
	size_t cursor = 24;
	for (int i = 0; i < metadata_kv_count; i++) {
		uint64_t key_len = consume_u64_little_endian(cursor);
		std::string key = this->consume_str(cursor, key_len);
		MetadataType type = static_cast<MetadataType>(this->consume_u32_little_endian(cursor));
		switch (type) {
			case MetadataType::str: {
				uint64_t str_len = consume_u64_little_endian(cursor);
				std::string value = this->consume_str(cursor, str_len);
				this->metadata.try_emplace(key, MetadataValue{type, value});
				break;
			}
			case MetadataType::u32: {
				uint32_t value = this->consume_u32_little_endian(cursor);
				this->metadata.try_emplace(key, MetadataValue{type, value});
				break;
			}
			case MetadataType::u64: {
				uint64_t value = this->consume_u32_little_endian(cursor);
				this->metadata.try_emplace(key, MetadataValue{type, value});
				break;
			}
			case MetadataType::f32: {
				float value = this->consume_f32_little_endian(cursor);
				this->metadata.try_emplace(key, MetadataValue{type, value});
				break;
			}
			case MetadataType::boolean: {
				bool value = this->consume_bool(cursor);
				this->metadata.try_emplace(key, MetadataValue{type, value});	
				break;	
			}
			case MetadataType::array: {
				std::vector<MetadataValue> array_vals = this->consume_array(cursor);	
				this->metadata.try_emplace(key, MetadataValue{type, array_vals});
				break;
			}
			default: {
				throw std::invalid_argument("got invalid metadata value type: " + std::to_string(type));
			}
		}
	}
	return cursor;
}

size_t WeightLoader::parse_tensor_info(size_t cursor) {
	for (int i = 0; i < this->parse_tensor_count(); i++) {
		TensorInfo t;
		uint64_t str_len = consume_u64_little_endian(cursor);
		t.name = this->consume_str(cursor, str_len);
		t.n_dim = this->consume_u32_little_endian(cursor);
		t.dim.resize(t.n_dim);
		for (int j = 0; j < t.n_dim; j++) {
			t.dim[j] = this->consume_u64_little_endian(cursor);
		}
		t.type = static_cast<TensorType>(this->consume_u32_little_endian(cursor));
		t.offset = this->consume_u64_little_endian(cursor);
		// this->tensor_info.push_back(t);
		this->tensor_index.try_emplace(t.name, t);
	}
	return cursor;
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

bool WeightLoader::consume_bool(size_t& cursor) {
	if (this->buffer[cursor] == 0) {
		cursor++;
		return false;
	} else if (this->buffer[cursor] == 1) {
		cursor++;
		return true;
	}
	throw std::invalid_argument("invalid bool");
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

float WeightLoader::consume_f32_little_endian(size_t& cursor) {
	float result = std::bit_cast<float>(consume_u32_little_endian(cursor));
	return result;
}

int32_t WeightLoader::consume_i32_little_endian(size_t& cursor) {
	int32_t result = std::bit_cast<int32_t>(consume_u32_little_endian(cursor));
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
	switch (arr_type) {
		case MetadataType::str: {
			for (size_t i = 0; i < arr_len; i++) {
				uint64_t str_len = this->consume_u64_little_endian(cursor);
				std::string str = this->consume_str(cursor, str_len);
				arr_vals.push_back(MetadataValue{arr_type, str});
			}
			break;
		}
		case MetadataType::u32: {
			for (size_t i = 0; i < arr_len; i++) {
				uint32_t val = this->consume_u32_little_endian(cursor);
				arr_vals.push_back(MetadataValue{arr_type, val});
			}
			break;
		}
		case MetadataType::u64: {
			for (size_t i = 0; i < arr_len; i++) {
				uint64_t val = this->consume_u64_little_endian(cursor);
				arr_vals.push_back(MetadataValue{arr_type, val});
			}
			break;
		}
		case MetadataType::f32: {
			for (size_t i = 0; i < arr_len; i++) {
				float val = this->consume_f32_little_endian(cursor);
				arr_vals.push_back(MetadataValue{arr_type, val});
			}
			break;
		}
		case MetadataType::i32: {
			for (size_t i = 0; i < arr_len; i++) {
				int32_t val = this->consume_i32_little_endian(cursor);
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


static void dump_metadata_value(const MetadataValue& mv, int indent = 0) {
    auto pad = std::string(indent, ' ');

    std::visit([&](const auto& v) {
        using T = std::decay_t<decltype(v)>;

        if constexpr (std::is_same_v<T, std::vector<MetadataValue>>) {
            std::cout << pad << "[array size=" << v.size() << "]\n";
            for (const auto& elem : v) {
                dump_metadata_value(elem, indent + 2);
            }
        } else if constexpr (std::is_same_v<T, uint8_t> ||
                             std::is_same_v<T, int8_t>) {
            std::cout << pad << static_cast<int>(v) << "\n";
        } else if constexpr (std::is_same_v<T, bool>) {
            std::cout << pad << (v ? "true" : "false") << "\n";
        } else {
            std::cout << pad << v << "\n";
        }
    }, mv.payload);
}

void WeightLoader::dump_metadata() {
	std::cout << "[   METADATA   ]\n";
	for (auto &kv : this->metadata) {
		std::cout << kv.first << "  --> \n\t";
		dump_metadata_value(kv.second, 2);
		std::cout << "\n";
	}
}

void WeightLoader::dump_tensor_info() {
	std::cout << "[   TENSOR INFO   ]\n";
	for (auto &kv : this->tensor_index) {
		auto t = kv.second;
		std::cout << "name: " << t.name << "\n";
		std::cout << "type: " << t.type << "\n";
		std::cout << "\tn_dim: " << t.n_dim << "\n";
		std::cout << "\tdim: ";
		for (int i = 0; i < t.n_dim; i++) {
			std::cout << t.dim[i] << " ";
		}
		std::cout << "\n";
		std::cout << "\toffset: " << t.offset << "\n";
	}
}

void WeightLoader::dump_tensor_view(TensorView t) {
	std::cout << "[   TENSOR VIEW   ]\n";
	std::cout << t.name << "\n";
	std::cout << "\t" << t.n_bytes << "bytes\n";
	for (int i = 0; i < t.dim.size(); i++) {
		std::cout << t.dim[i] << " ";
	}
	std::cout << "\n";
}

