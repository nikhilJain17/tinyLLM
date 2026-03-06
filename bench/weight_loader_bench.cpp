#include "weightloader/weightloader.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <mach/mach.h>
#include <random>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>

struct PageFaults {
    uint64_t major = 0;
    uint64_t minor = 0;
};

struct Snapshot {
    PageFaults pf;
    uint64_t rss_bytes = 0;
};

struct TelemetryRow {
    size_t tensor_idx = 0;
    std::string tensor_name;
    uint64_t major_faults_delta = 0;
    uint64_t minor_faults_delta = 0;
    int64_t rss_delta_bytes = 0;
    uint64_t rss_bytes = 0;
};

enum class MeasureMode {
    Latency,
    Telemetry
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
    return PageFaults{
        static_cast<uint64_t>(ru.ru_majflt),
        static_cast<uint64_t>(ru.ru_minflt),
    };
}

uint64_t get_rss_bytes() {
    mach_task_basic_info info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;

    if (task_info(mach_task_self(),
                  MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info),
                  &count) != KERN_SUCCESS) {
        return 0;
    }

    return static_cast<uint64_t>(info.resident_size);
}

Snapshot take_snapshot() {
    return Snapshot{get_self_page_faults(), get_rss_bytes()};
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

std::string mode_to_string(Mode mode) {
    switch (mode) {
    case Mode::MemoryMapped:
        return "MemoryMapped";
    case Mode::ChunkedFullyResident:
        return "ChunkedFullyResident";
    default:
        return "UnknownMode";
    }
}

std::string workload_test_name(const std::string &workload) {
    if (workload == "sequential") return "SEQUENTIAL_WORKLOAD";
    if (workload == "layer") return "LAYER_BY_LAYER_WORKLOAD";
    if (workload == "random") return "RANDOM_WORKLOAD";
    throw std::runtime_error("unknown workload: " + workload);
}

void print_latency_csv_header() {
    std::cout << "test,model,mode,cold_hot,latency_ms,major_faults_delta,minor_faults_delta\n";
}

void print_latency_csv_row(const std::string &test_name,
                           const std::string &model_filepath,
                           Mode mode,
                           const std::string &cold_hot,
                           std::chrono::milliseconds latency,
                           const PageFaults &before,
                           const PageFaults &after) {
    PageFaults delta = after - before;

    std::cout << test_name << ","
              << model_filepath << ","
              << mode_to_string(mode) << ","
              << cold_hot << ","
              << latency.count() << ","
              << delta.major << ","
              << delta.minor << "\n";
}

void print_telemetry_csv_header() {
    std::cout << "test,model,mode,cold_hot,tensor_idx,tensor_name,"
                 "major_faults_delta,minor_faults_delta,"
                 "rss_delta_bytes,rss_bytes\n";
}

void print_telemetry_csv_rows(const std::string &test_name,
                              const std::string &model_filepath,
                              Mode mode,
                              const std::string &cold_hot,
                              const std::vector<TelemetryRow> &rows) {
    for (const auto &row : rows) {
        std::cout << test_name << ","
                  << model_filepath << ","
                  << mode_to_string(mode) << ","
                  << cold_hot << ","
                  << row.tensor_idx << ","
                  << row.tensor_name << ","
                  << row.major_faults_delta << ","
                  << row.minor_faults_delta << ","
                  << row.rss_delta_bytes << ","
                  << row.rss_bytes << "\n";
    }
}

void access_tensor(WeightLoader &w, Mode mode, const std::string &tensor_name) {
    auto r = w.fetch_tensor(tensor_name);
    assert(r.has_value());

    if (mode == Mode::MemoryMapped) {
        w.mmap_load_tensor_into_memory(r->data, r->n_bytes);
    }
}

std::vector<std::string> workload_tensor_names(const std::string &tensor_name_filepath,
                                               const std::string &workload) {
    auto names = load_tensor_names(tensor_name_filepath);

    if (workload == "random") {
        std::mt19937 rng(100);
        std::shuffle(names.begin(), names.end(), rng);
    }

    // sequential + layer preserve the order in the file
    return names;
}

void run_latency_workload(const std::string &test_name,
                          const std::string &model_filepath,
                          const std::vector<std::string> &tensor_names,
                          Mode mode,
                          const std::string &cold_hot) {
    using clock = std::chrono::steady_clock;

    PageFaults before = get_self_page_faults();
    auto start = clock::now();

    // Includes constructor/load phase.
    WeightLoader w(model_filepath, mode);

    // Includes access phase.
    for (const auto &tensor_name : tensor_names) {
        access_tensor(w, mode, tensor_name);
    }

    auto end = clock::now();
    PageFaults after = get_self_page_faults();

    auto latency =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    print_latency_csv_row(test_name, model_filepath, mode, cold_hot, latency, before, after);
}

void run_telemetry_workload(const std::string &test_name,
                            const std::string &model_filepath,
                            const std::vector<std::string> &tensor_names,
                            Mode mode,
                            const std::string &cold_hot) {
    std::vector<TelemetryRow> rows;

    if (mode == Mode::ChunkedFullyResident) {
        // For fully resident mode, the meaningful memory activity is in the constructor.
        Snapshot before = take_snapshot();
        WeightLoader w(model_filepath, mode);
        Snapshot after = take_snapshot();
        PageFaults delta = after.pf - before.pf;

        rows.push_back(TelemetryRow{
            0,
            "CONSTRUCTOR_LOAD",
            delta.major,
            delta.minor,
            static_cast<int64_t>(after.rss_bytes) - static_cast<int64_t>(before.rss_bytes),
            after.rss_bytes,
        });

        // Optional traversal, but we intentionally do not log per-tensor because
        // the interesting faults already happened during the constructor load.
        for (const auto &tensor_name : tensor_names) {
            auto r = w.fetch_tensor(tensor_name);
            assert(r.has_value());
        }

        print_telemetry_csv_rows(test_name, model_filepath, mode, cold_hot, rows);
        return;
    }

    // MemoryMapped mode: constructor sets up mapping, per-tensor access is the interesting part.
    WeightLoader w(model_filepath, mode);
    rows.reserve(tensor_names.size());

    Snapshot prev = take_snapshot();

    for (size_t i = 0; i < tensor_names.size(); i++) {
        const auto &tensor_name = tensor_names[i];

        access_tensor(w, mode, tensor_name);

        Snapshot curr = take_snapshot();
        PageFaults delta = curr.pf - prev.pf;

        rows.push_back(TelemetryRow{
            i,
            tensor_name,
            delta.major,
            delta.minor,
            static_cast<int64_t>(curr.rss_bytes) - static_cast<int64_t>(prev.rss_bytes),
            curr.rss_bytes,
        });

        prev = curr;
    }

    print_telemetry_csv_rows(test_name, model_filepath, mode, cold_hot, rows);
}

void run_workload(const std::string &workload,
                  const std::string &model_filepath,
                  const std::string &tensor_name_filepath,
                  Mode mode,
                  MeasureMode measure_mode,
                  const std::string &cold_hot) {
    auto tensor_names = workload_tensor_names(tensor_name_filepath, workload);
    auto test_name = workload_test_name(workload);

    if (measure_mode == MeasureMode::Latency) {
        run_latency_workload(test_name, model_filepath, tensor_names, mode, cold_hot);
    } else {
        run_telemetry_workload(test_name, model_filepath, tensor_names, mode, cold_hot);
    }
}

[[noreturn]] void usage(const char *prog) {
    std::cerr
        << "Usage:\n"
        << "  " << prog
        << " --model <gguf>"
        << " --tensor-names <txt>"
        << " --mode <mmap|fully_resident>"
        << " --workload <sequential|layer|random>"
        << " --measure <latency|telemetry>"
        << " --cache-state <cold|hot>\n";
    std::exit(1);
}

int main(int argc, char **argv) {
    std::string model;
    std::string tensor_names_file;
    std::string mode_str;
    std::string workload_str;
    std::string measure_str;
    std::string cold_hot;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (arg == "--tensor-names" && i + 1 < argc) {
            tensor_names_file = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode_str = argv[++i];
        } else if (arg == "--workload" && i + 1 < argc) {
            workload_str = argv[++i];
        } else if (arg == "--measure" && i + 1 < argc) {
            measure_str = argv[++i];
        } else if (arg == "--cache-state" && i + 1 < argc) {
            cold_hot = argv[++i];
        } else {
            usage(argv[0]);
        }
    }

    if (model.empty() || tensor_names_file.empty() || mode_str.empty() ||
        workload_str.empty() || measure_str.empty() || cold_hot.empty()) {
        usage(argv[0]);
    }

    Mode mode;
    if (mode_str == "mmap") {
        mode = Mode::MemoryMapped;
    } else if (mode_str == "fully_resident") {
        mode = Mode::ChunkedFullyResident;
    } else {
        usage(argv[0]);
    }

    MeasureMode measure_mode;
    if (measure_str == "latency") {
        measure_mode = MeasureMode::Latency;
        print_latency_csv_header();
    } else if (measure_str == "telemetry") {
        measure_mode = MeasureMode::Telemetry;
        print_telemetry_csv_header();
    } else {
        usage(argv[0]);
    }

    run_workload(workload_str, model, tensor_names_file, mode, measure_mode, cold_hot);
    return 0;
}