#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <limits>

#include "dot_product.h"

constexpr float EPS = std::numeric_limits<float>::epsilon();

bool almost_equal(float a, float b,
                  float rel_tol = 10.0f * EPS,
                  float abs_tol = 10.0f * EPS)
{
    float diff = std::fabs(a - b);
    float norm  = std::fmax(std::fabs(a), std::fabs(b));
    return diff <= std::fmax(rel_tol * norm, abs_tol);
}

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

void benchmark(const std::vector<float> &A, const std::vector<float> &B, const int warmup, const int measured_iterations, std::pair<float, double> (*dot_product)(const std::vector<float> &a, const std::vector<float> &b)) {
    const int total_runs = warmup + measured_iterations;

     std::vector<double> kernel_runtimes;
    kernel_runtimes.reserve(measured_iterations); 
    
    std::vector<double> wall_clock_times;
    wall_clock_times.reserve(measured_iterations);

    for (int run = 0; run < total_runs; ++run) {
        // Wall clock time
        auto start = std::chrono::high_resolution_clock::now();
        
        auto [result, time] = dot_product(A, B);

        auto end = std::chrono::high_resolution_clock::now();

        if (run >= warmup) {
            wall_clock_times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
            kernel_runtimes.push_back(time); // Time of cpu reduce + gpu kernel runtime
        }
    }

    auto [result, time] = dot_product(A, B);
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

    const std::uint64_t flops = 2 * A.size();          // multiply-add
    double medianGflops = flops / median_kernel_runtimes;   // 1 ns = 1e-9 s
    double meanGflops   = flops / mean_kernel_runtimes;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Median kernel time : " << median_kernel_runtimes / 1e6 << " ms\n";
    std::cout << "Mean   kernel time : " << mean_kernel_runtimes   / 1e6 << " ms\n";
    std::cout << "Median throughput  : " << medianGflops << " GFLOP/s\n";
    std::cout << "Mean   throughput  : " << meanGflops   << " GFLOP/s\n";

    std::cout << "Median wall clock time: " << median_wall_clock_times / 1e6 << "ms\n";
    std::cout << "Mean wall clock time: " << mean_wall_clock_times / 1e6 << "ms\n";
}

void test(const std::vector<float> &A, const std::vector<float> &B, std::pair<float, double> (*dot_product)(const std::vector<float> &a, const std::vector<float> &b)) {
    float cpu_result = 0;
    #pragma omp parallel for reduction(+:cpu_result)
    for (size_t i = 0; i < A.size(); i++) {
        cpu_result += A[i] * B[i];
    }
    auto [result, time] = dot_product(A, B);

    std::cout << cpu_result << " " << result << std::endl;

    assert(almost_equal(cpu_result, result));
}

int main() {
    const size_t N = 1024;
    const int warmup = 5;
    const int measured_iterations = 100;

    std::vector<float> A = random_vector<float>(N);
    std::vector<float> B = random_vector<float>(N);

    @autoreleasepool { // Clean objective-c objects
        std::cout << "Dot product mul, cpu reduce\n" << std::endl;
        test(A, B, dot_product_mul);
        benchmark(A, B, warmup, measured_iterations, dot_product_mul);

        std::cout << "\nDot product mul reduce, cpu reduce\n" << std::endl;
        test(A, B, dot_product_mul_reduce);
        benchmark(A, B, warmup, measured_iterations, dot_product_mul_reduce);

        std::cout << "\nDot product mul reduce atomic\n" << std::endl;
        test(A, B, dot_product_mul_reduce_atomic);
        benchmark(A, B, warmup, measured_iterations, dot_product_mul_reduce_atomic);

        std::cout << "\nDot product mul tree reduce\n" << std::endl;
        test(A, B, dot_product_mul_tree_reduce);
        benchmark(A, B, warmup, measured_iterations, dot_product_mul_tree_reduce);
    }
}