#include "weightloader.hpp"

std::ostream& operator<<(std::ostream& out, const Mode& m) {
    if (m == Mode::FullyResidentOneShot) {
        out << "FullyResidentOneShot";
    } else if (m == Mode::ChunkedFullyResident) {
        out << "ChunkedFullyResident";
    } else if (m == Mode::MemoryMapped) {
        out << "MemoryMapped";
    }
    return out;
}

WeightLoader::WeightLoader(const std::string &filepath, Mode mode) : filepath(filepath) {
	this->fully_resident = (mode == Mode::FullyResidentOneShot 
		|| mode == Mode::ChunkedFullyResident);
	if (!fully_resident) {
		this->load_mmap();
	} else {
		// Try fully resident one shot if requested
		// Otherwise, load in chunked
		if (mode == Mode::FullyResidentOneShot) {
			try {
				this->load_fully_resident();
			} catch (const ReadError& e) {
				DEBUG_LOG("Could not load fully resident one shot, falling back to fully resident chunked.");
				this->load_fully_resident_chunked();
			}
		} else {
			this->load_fully_resident_chunked();
		}
		// TODO: find out threshold for loading chunked
		// this->load_fully_resident();
	}

	DEBUG_LOG("Magic number is ",
			this->parse_magic_number() ? "VALID" : "NOT VALID");
	DEBUG_LOG("GGUF version ", this->parse_gguf_version());
	DEBUG_LOG("Contains ", this->parse_tensor_count(), " tensors.");
	DEBUG_LOG("Contains ", this->parse_metadata_kv_count(), " metadata kv pairs.");
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
}

WeightLoader::~WeightLoader() {
	if (!fully_resident) {
		munmap(this->mmap_ptr, this->buf_size);
	}
}

void WeightLoader::load_mmap() {
	DEBUG_LOG("Memory mapping weights from ", this->filepath);
	int fd = open(this->filepath.c_str(), O_RDONLY);
	struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error("fstat failed");
    }
    this->buf_size = static_cast<size_t>(st.st_size);
	this->mmap_ptr = static_cast<uint8_t*>(mmap(NULL, this->buf_size, PROT_READ, MAP_PRIVATE, fd, 0));
	if (this->mmap_ptr == MAP_FAILED) {
		throw std::runtime_error("mmap failed");
	}
}

void WeightLoader::load_fully_resident() {
	DEBUG_LOG("Loading weights from ", this->filepath);
	int fd = open(this->filepath.c_str(), O_RDONLY);
	if (fd == -1) {
		DEBUG_LOG("Error opening file! ", this->filepath);
		exit(-1);
	}
	struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error("fstat failed");
    }
    this->buf_size = static_cast<size_t>(st.st_size);
    this->buffer = std::make_unique<uint8_t[]>(this->buf_size);
    size_t total = 0;
    while (total < this->buf_size) {
        size_t remaining = this->buf_size - total;
        ssize_t n = read(fd, this->buffer.get() + total, remaining);

        if (n == 0) { // EOF unexpectedly early
            break;
        }
        if (n < 0) {
            if (errno == EINTR) continue; // interrupted, retry
            close(fd);
            throw ReadError("read failed");
        }

        total += static_cast<size_t>(n);
        DEBUG_LOG("Read ", total, "/", this->buf_size, " bytes (",
                  (100.0 * total) / this->buf_size, "%)");
    }

    close(fd);

    if (total != this->buf_size) {
        throw std::runtime_error("short read: file truncated?");
    }

    DEBUG_LOG("Successfully loaded weights totaling ", total, " bytes.");
}


void WeightLoader::load_fully_resident_chunked(const size_t CHUNK) {
    DEBUG_LOG("Loading weights from ", filepath);

    int fd = open(filepath.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error(std::string("open failed: ") + std::strerror(errno));
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        close(fd);
        throw std::runtime_error(std::string("fstat failed: ") + std::strerror(errno));
    }

    buf_size = static_cast<size_t>(st.st_size);
    buffer = std::make_unique<uint8_t[]>(buf_size);

    size_t total = 0;
    while (total < buf_size) {
        size_t remaining = buf_size - total;
        size_t to_read = std::min(remaining, CHUNK);

        ssize_t n = read(fd, buffer.get() + total, to_read);
        if (n == 0) break; // EOF early
        if (n < 0) {
            if (errno == EINTR) continue;
            int e = errno;
            close(fd);
            throw std::runtime_error(std::string("read failed: ") + std::strerror(e));
        }
        total += static_cast<size_t>(n);
    }

    close(fd);

    if (total != buf_size) {
        throw std::runtime_error("short read: expected " + std::to_string(buf_size) +
                                 " got " + std::to_string(total));
    }

    DEBUG_LOG("Successfully loaded ", total, " bytes.");
}

std::unordered_map<std::string, TensorInfo> WeightLoader::get_tensor_index() {
	return this->tensor_index;
}

void WeightLoader::mmap_load_tensor_into_memory(const uint8_t* offset, size_t n_bytes) {
	volatile int x = 0;
	for (int i = 0; i < n_bytes; i += this->PAGE_SIZE) {
		x += offset[i];
	}
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
	t.data = this->base_ptr() + info.offset;
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
			throw std::invalid_argument("unsupported tensor type: " + std::to_string(t.type));
	}
	t.dim = std::span<const uint64_t>(info.dim.data(), info.dim.size());

	return t;
}


bool WeightLoader::parse_magic_number() {
	return this->base_ptr()[0] == 0x47 && this->base_ptr()[1] == 0x47 &&
		   this->base_ptr()[2] == 0x55 && this->base_ptr()[3] == 0x46;
}

int WeightLoader::parse_gguf_version() {
	std::cout << "Parse gguf version:";
	uint32_t gguf_version = 0;
	for (int i = 0; i < 4; i++) {
		gguf_version += this->base_ptr()[4 + i] << (8 * i);
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
		this->tensor_index.try_emplace(t.name, t);
	}
	return cursor;
}

uint64_t WeightLoader::peek_u64_little_endian(size_t index) {
	if (index + 7 >= this->buf_size) {
		throw std::invalid_argument("trying to read beyond buffer size!");
	}
	uint64_t result = 0;
	for (int i = 0; i < 8; i++) {
		result += uint64_t(this->base_ptr()[index + i]) << (8 * i);
	}
	return result;
}

uint32_t WeightLoader::peek_u32_little_endian(size_t index) {
	if (index + 4 >= this->buf_size) {
		throw std::invalid_argument("trying to read beyond buffer size!");
	}
	uint32_t result = 0;
	for (int i = 0; i < 4; i++) {
		result += uint32_t(this->base_ptr()[index + i]) << (8 * i);
	}
	return result;
}

bool WeightLoader::consume_bool(size_t& cursor) {
	if (this->base_ptr()[cursor] == 0) {
		cursor++;
		return false;
	} else if (this->base_ptr()[cursor] == 1) {
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
    const char* p = reinterpret_cast<const char*>(this->base_ptr() + cursor);
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
		std::cout << "- name: " << t.name << "\n";
		std::cout << "\ttype: " << t.type << "\n";
		std::cout << "\tn_dim: " << t.n_dim << "\n";
		std::cout << "\tdim: ";
		for (int i = 0; i < t.n_dim; i++) {
			std::cout << t.dim[i] << " ";
		}
		std::cout << "\n";
		std::cout << "\toffset: " << t.offset << "\n\n";
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

