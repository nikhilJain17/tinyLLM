#include "weightloader/weightloader.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sys/resource.h>
#include <vector>

struct PageFaults {
	uint64_t major = 0;
	uint64_t minor = 0;
};

std::ostream &operator<<(std::ostream &out, const PageFaults &p) {
	out << "PageFaults(major=" << p.major << ", minor=" << p.minor << ")";
	return out;
}

PageFaults operator-(const PageFaults &a, const PageFaults &b) {
	return PageFaults{a.major - b.major, a.minor - b.minor};
}

PageFaults get_self_page_faults() {
	rusage ru{};
	getrusage(RUSAGE_SELF, &ru);
	return PageFaults{static_cast<uint64_t>(ru.ru_majflt),
					  static_cast<uint64_t>(ru.ru_minflt)};
}

void print_bench_result(const std::string &test_name,
						const std::string &model_filepath, const Mode &mode,
						std::chrono::steady_clock::time_point start,
						std::chrono::steady_clock::time_point end,
						const PageFaults &page_faults_before,
						const PageFaults &page_faults_after) {

	auto latency =
		std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

	std::cout << "[   " << test_name << "   ]\n"
			  << "\tWeights: " << model_filepath << "\n"
			  << "\tMode: " << mode << "\n"
			  << "\tE2E Latency: " << latency << "\n"
			  << "\tPage faults before: " << page_faults_before << "\n"
			  << "\tPage faults after: " << page_faults_after << "\n"
			  << "\tPage faults delta: "
			  << (page_faults_after - page_faults_before) << "\n";
}

std::vector<std::string> load_tensor_names(const std::string &path) {
	std::vector<std::string> names;
	std::ifstream file(path);

	if (!file) {
		throw std::runtime_error("Failed to open " + path);
	}

	std::string line;
	while (std::getline(file, line)) {
		if (!line.empty()) {
			names.push_back(line);
		}
	}

	return names;
}

void sequential_test(std::string model_filepath, Mode mode) {
	std::chrono::high_resolution_clock clock;
    PageFaults page_faults_before = get_self_page_faults();
	auto start = clock.now();
	WeightLoader w(model_filepath, mode);
	std::unordered_map<std::string, TensorInfo> tensor_index =
		w.get_tensor_index();
	for (auto &kv : tensor_index) {
		auto r = w.fetch_tensor(kv.first);
		if (mode == Mode::MemoryMapped) {
			w.mmap_load_tensor_into_memory(r.value().data, r.value().n_bytes);
		}
		assert(r.has_value());
	}
	auto end = clock.now();
    PageFaults page_faults_after = get_self_page_faults();
	print_bench_result("SEQUENTIAL WORKLOAD", model_filepath, mode, start, end,
					   page_faults_before, page_faults_after);
}

void random_test(std::string model_filepath, std::string tensor_name_filepath,
				 Mode mode) {
	std::mt19937 rng(100); // fixed seed for benchmarks
	std::vector<std::string> tensor_names =
		load_tensor_names(tensor_name_filepath);
	std::shuffle(tensor_names.begin(), tensor_names.end(), rng);
	std::chrono::high_resolution_clock clock;
	PageFaults page_faults_before = get_self_page_faults();
	auto start = clock.now();
	WeightLoader w(model_filepath, mode);
	for (auto t : tensor_names) {
		auto r = w.fetch_tensor(t);
		if (mode == Mode::MemoryMapped) {
			w.mmap_load_tensor_into_memory(r.value().data, r.value().n_bytes);
		}
	}
	auto end = clock.now();
	PageFaults page_faults_after = get_self_page_faults();
	print_bench_result("RANDOM WORKLOAD", model_filepath, mode, start, end,
					   page_faults_before, page_faults_after);
}

void layer_by_layer_test(std::string model_filepath,
						 std::string tensor_name_filepath, Mode mode) {
	std::vector<std::string> tensor_names =
		load_tensor_names(tensor_name_filepath);
	std::chrono::high_resolution_clock clock;
    PageFaults page_faults_before = get_self_page_faults();
	auto start = clock.now();
	WeightLoader w(model_filepath, mode);
	for (auto t : tensor_names) {
		auto r = w.fetch_tensor(t);
		if (mode == Mode::MemoryMapped) {
			w.mmap_load_tensor_into_memory(r.value().data, r.value().n_bytes);
		}
	}
	auto end = clock.now();
    PageFaults page_faults_after = get_self_page_faults();
	print_bench_result("LAYER BY LAYER WORKLOAD", model_filepath, mode, start, end,
					   page_faults_before, page_faults_after);
}

int main() {
    for (int i = 0; i < static_cast<int>(Mode::COUNT) * 10; i++) {
        sequential_test("resources/"
                        "ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-llama-"
                        "3.1-8b-instruct-q4_0.gguf",
                        static_cast<Mode>(i % 3));

        layer_by_layer_test("resources/"
                            "ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-"
                            "llama-3.1-8b-instruct-q4_0.gguf",
                            "bench/llama-3.1-tensor-names.txt", static_cast<Mode>(i % 3));
        random_test("resources/"
                    "ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-llama-3.1-"
                    "8b-instruct-q4_0.gguf",
                    "bench/llama-3.1-tensor-names.txt", static_cast<Mode>(i % 3));
        std::cout << "_________________________\n";
    }

}