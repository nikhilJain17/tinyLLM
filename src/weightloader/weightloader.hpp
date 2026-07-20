#pragma once

#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include <unordered_map>
#include <variant>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <span>
#include <optional>
#include <sys/mman.h>
#include <unistd.h>


typedef struct {
	int num_tensors;

} Metadata;

enum TensorType {
	GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    GGML_TYPE_Q4_0    = 2,
    GGML_TYPE_Q4_1    = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0    = 6,
    GGML_TYPE_Q5_1    = 7,
    GGML_TYPE_Q8_0    = 8,
    GGML_TYPE_Q8_1    = 9,
    GGML_TYPE_Q2_K    = 10,
    GGML_TYPE_Q3_K    = 11,
    GGML_TYPE_Q4_K    = 12,
    GGML_TYPE_Q5_K    = 13,
    GGML_TYPE_Q6_K    = 14,
    GGML_TYPE_Q8_K    = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
    // GGML_TYPE_Q4_0_4_8 = 32,
    // GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
    // GGML_TYPE_IQ4_NL_4_4 = 36,
    // GGML_TYPE_IQ4_NL_4_8 = 37,
    // GGML_TYPE_IQ4_NL_8_8 = 38,
    GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
    GGML_TYPE_COUNT   = 40,
};

typedef struct {
	std::string name;
	TensorType type;
	uint64_t offset;
	uint32_t n_dim;
	std::vector<uint64_t> dim;
} TensorInfo;

typedef struct {
	std::string name;
	const uint8_t* data;
	size_t n_bytes;
	TensorType type;
	std::span<const uint64_t> dim;
} TensorView;

// From https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
enum MetadataType: uint32_t {
    // The value is a 8-bit unsigned integer.
    u8 = 0,
    // The value is a 8-bit signed integer.
    i8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    u16 = 2,
    // The value is a 16-bit signed little-endian integer.
    i16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    u32 = 4,
    // The value is a 32-bit signed little-endian integer.
    i32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    f32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    boolean = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    str = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of bytes.
    array = 9,
    // The value is a 64-bit unsigned little-endian integer.
    u64 = 10,
    // The value is a 64-bit signed little-endian integer.
    i64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    f64 = 12,
};

struct MetadataValue;

using MetadataPayload = std::variant<
    uint8_t,
    int8_t,
    uint16_t,
    int16_t,
    uint32_t,
    int32_t,
    float,
    bool,
    std::string,
    std::vector<MetadataValue>,	
    uint64_t,
    int64_t,
    double
>;


struct MetadataValue {
	MetadataType type;
	MetadataPayload payload;
};


enum class Mode {
    FullyResidentOneShot = 0,
    ChunkedFullyResident,
    MemoryMapped,
    COUNT
};

struct ReadError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

std::ostream& operator<<(std::ostream& out, const Mode& m) ;

using MetadataValue = struct MetadataValue;

class WeightLoader {
  private:
	bool fully_resident;
	std::unique_ptr<uint8_t[]> buffer;
	uint8_t* mmap_ptr;
	const std::string filepath;	
	size_t buf_size;
	const size_t PAGE_SIZE = static_cast<size_t>(sysconf(_SC_PAGESIZE));
 
	// TODO: Turn metadata and tensor_info into zero-copy structs?
	std::unordered_map<std::string, MetadataValue> metadata;
	// std::vector<TensorInfo> tensor_info;
	std::unordered_map<std::string, TensorInfo> tensor_index; // name --> offset

	uint8_t* base_ptr() {
		return fully_resident ? buffer.get() : mmap_ptr; 
	}

	// Parse specific parts of the gguf file
	bool parse_magic_number();
	int parse_gguf_version();
	uint64_t parse_tensor_count();
	uint64_t parse_metadata_kv_count();
	size_t parse_metadata_kv_pairs();
	size_t parse_tensor_info(size_t);

	// Debug functions
	void dump_metadata();
	void dump_tensor_info();
	void dump_tensor_view(TensorView);
	
	// Helper functions to traverse file
	uint64_t peek_u64_little_endian(size_t);
	uint32_t peek_u32_little_endian(size_t);
	uint64_t consume_u64_little_endian(size_t&);
	uint32_t consume_u32_little_endian(size_t&);
	float consume_f32_little_endian(size_t&);
	int32_t consume_i32_little_endian(size_t&);
	bool consume_bool(size_t&);
	std::string consume_str(size_t&, size_t);
	std::vector<MetadataValue> consume_array(size_t&);
	
	// If the GPU has enough VRAM, load all tensors
	void load_fully_resident();
	void load_fully_resident_chunked(const size_t CHUNK=64ull * 1024 * 1024);
	// Otherwise, stream them from disk through memory to GPU
	void load_mmap();
  public:
	WeightLoader(const std::string &, Mode);
	~WeightLoader();
	Metadata get_metadata();
    void write_token_vocab_to_file(std::string filepath);
    void write_token_merges_to_file(std::string filepath);
	std::unordered_map<std::string, TensorInfo> get_tensor_index();
	std::optional<TensorView> fetch_tensor(std::string_view tensor_name);
	void mmap_load_tensor_into_memory(const uint8_t*, size_t);
};

// To convert pointer from mmap to unique_ptr
struct MMapDeleter {
    size_t size;
    void operator()(uint8_t* p) const {
        if (p) munmap(p, size);
    }
};