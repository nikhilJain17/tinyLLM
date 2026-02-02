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

#define CHUNK 16777216

typedef struct {
	int num_tensors;

} Metadata;

typedef struct {
	std::vector<int> dim;
	std::vector<uint8_t> data;
} Tensor;

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

using MetadataValue = struct MetadataValue;


class WeightLoader {
  private:
	std::unique_ptr<uint8_t[]> buffer;
	const std::string filepath;	
	int buf_size;
	std::unordered_map<std::string, MetadataValue> metadata;

	bool parse_magic_number();
	int parse_gguf_version();
	uint64_t parse_tensor_count();
	uint64_t parse_metadata_kv_count();
	void parse_metadata_kv_pairs();
	void dump_metadata();
	
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
	// Otherwise, stream them to GPU
	bool stream_to_gpu();
  public:
	WeightLoader(const std::string &filepath);
	Metadata get_metadata();
	Tensor fetch_tensor(std::string &tensor_name);
};
