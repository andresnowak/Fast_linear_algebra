#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "dot_product.h"

void print_vector(const std::vector<float> &a) {
    for (const auto& x : a) {
        std::cout << x << ", ";
    }

    std::cout << std::endl;
}

template <typename T>
std::vector<T> random_vector(size_t n, float low = -10, float high = 10) {
    std::mt19937 rng(41);
    std::uniform_int_distribution<int> dist(low, high);
    std::vector<T> v(n);
    for (auto &x : v) {
        x = (float)dist(rng);
    }

    return v;
}

int main() {
    const size_t N = 1048576;
    const int warmup = 5;
    const int measured_iterations = 100;
    const int total_runs = warmup + measured_iterations;

    std::vector<float> A = random_vector<float>(N);
    std::vector<float> B = random_vector<float>(N);

    std::vector<double> kernel_runtimes;
    kernel_runtimes.reserve(measured_iterations); 
    
    std::vector<double> wall_clock_times;
    wall_clock_times.reserve(measured_iterations);

    for (int run = 0; run < total_runs; ++run) {
        @autoreleasepool { // Clean objective-c objects
            // Wall clock time
            auto start = std::chrono::high_resolution_clock::now();
            
            auto [result, time] = dot_product_mul(A, B);

            auto end = std::chrono::high_resolution_clock::now();

            if (run >= warmup) {
                wall_clock_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                kernel_runtimes.push_back(time); // Time of cpu reduce + gpu kernel runtime
            }
        }
    }

    auto [result, time] = dot_product_mul(A, B);
    std::cout << "Dot product result: " << result << std::endl;

    // Get statistics
    std::sort(kernel_runtimes.begin(), kernel_runtimes.end());
    std::sort(wall_clock_times.begin(), wall_clock_times.end());

    double median_kernel_runtimes = 0;
    double median_wall_clock_times = 0;
    if (measured_iterations % 2 == 0) {
        median_kernel_runtimes = (kernel_runtimes[measured_iterations / 2] + kernel_runtimes[measured_iterations / 2 + 1]) / 2;
        median_wall_clock_times = (wall_clock_times[measured_iterations / 2] + wall_clock_times[measured_iterations / 2 + 1]) / 2;
    } else {
        median_kernel_runtimes = kernel_runtimes[(measured_iterations + 1) / 2];
        median_wall_clock_times = wall_clock_times[(measured_iterations + 1) / 2];
    }

    double mean_kernel_runtimes = std::accumulate(kernel_runtimes.begin(), kernel_runtimes.end(), 0.0) / kernel_runtimes.size();
    double mean_wall_clock_times = std::accumulate(wall_clock_times.begin(), wall_clock_times.end(), 0.0) / wall_clock_times.size();


    const std::uint64_t flops = 2 * N;          // multiply-add
    double medianGflops = flops / median_kernel_runtimes;   // 1 ns = 1e-9 s
    double meanGflops   = flops / mean_kernel_runtimes;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Median kernel time : " << median_kernel_runtimes / 1e6 << " ms\n";
    std::cout << "Mean   kernel time : " << mean_kernel_runtimes   / 1e6 << " ms\n";
    std::cout << "Median throughput  : " << medianGflops << " GFLOP/s\n";
    std::cout << "Mean   throughput  : " << meanGflops   << " GFLOP/s\n";

    std::cout << "Median wall clock time: " << median_wall_clock_times / 1e6 << "ms\n";
    std::cout << "Mean wall clock time: " << mean_wall_clock_times / 1e6 << "ms\n";

    return 0;
}