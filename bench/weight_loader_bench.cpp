#include "weightloader/weightloader.hpp"

#include <chrono>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

std::vector<std::string> load_tensor_names(const std::string& path) {
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

void sequential_test(std::string model_filepath) {
    std::chrono::high_resolution_clock clock;
    auto start = clock.now();
	WeightLoader w(
        "resources/ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-llama-3.1-8b-instruct-q4_0.gguf");
    std::unordered_map<std::string, TensorInfo> tensor_index = w.get_tensor_index();
    for (auto &kv : tensor_index) {
        w.fetch_tensor(kv.first);
    }
    auto end = clock.now();
    std::cout << "[   SEQUENTIAL TEST   ]\n"
            << "\t- " << model_filepath << "\n"
            << "\t- " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    end - start)
            << "\n";
}

void random_test(std::string model_filepath, std::string tensor_name_filepath) {
    std::mt19937 rng(100); // fixed seed for benchmarks
    std::vector<std::string> tensor_names = load_tensor_names(tensor_name_filepath);
    std::shuffle(tensor_names.begin(), tensor_names.end(), rng);
    std::chrono::high_resolution_clock clock;
    auto start = clock.now();
    WeightLoader w(model_filepath);    
    for (auto t : tensor_names) {
        w.fetch_tensor(t);
    }
    auto end = clock.now();
    std::cout << "[   RANDOM TEST   ]\n"
            << "\t- " << model_filepath << "\n"
            << "\t- " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    end - start)
            << "\n";
}


void layer_by_layer_test(std::string model_filepath, std::string tensor_name_filepath) {
    std::vector<std::string> tensor_names = load_tensor_names(tensor_name_filepath);
    std::chrono::high_resolution_clock clock;
    auto start = clock.now();
    WeightLoader w(model_filepath);
    for (auto t : tensor_names) {
        w.fetch_tensor(t);
    }
    auto end = clock.now();
    std::cout << "[   PER-LAYER TEST   ]\n"
            << "\t- " << model_filepath << "\n"
            << "\t- " << std::chrono::duration_cast<std::chrono::milliseconds>(
                    end - start)
            << "\n";
}

int main() {
    sequential_test(
        "resources/ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-llama-3.1-8b-instruct-q4_0.gguf");
    layer_by_layer_test("resources/ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-llama-3.1-8b-instruct-q4_0.gguf",
        "bench/llama-3.1-tensor-names.txt");
    random_test("resources/ggml-org_Meta-Llama-3.1-8B-Instruct-Q4_0-GGUF_meta-llama-3.1-8b-instruct-q4_0.gguf",
        "bench/llama-3.1-tensor-names.txt");
}