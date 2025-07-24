#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "dot_product.h"

void print_vector(const std::vector<float> &a) {
    for (const auto& x : a) {
        std::cout << x << ", ";
    }

    std::cout << std::endl;
}

int main() {
    const std::size_t n = 1048576;
    std::mt19937 rng(41);
    std::uniform_int_distribution<int> dist(-10.0f, 10.0f);
    
    std::vector<float> A(n), B(n);
    for (std::size_t i = 0; i < n; i++) {
        A[i] = (float)dist(rng);
        B[i] = (float)dist(rng);
    }

    auto start = std::chrono::high_resolution_clock::now();
    auto result_time = dot_product_mul(A, B);
    auto result = result_time.first;
    auto time = result_time.second;
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Dot product result: " << result << std::endl;

    int total_flops = 2 * n;
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    double gflops = static_cast<double>(total_flops) / time;   // GFLOP/s

    std::cout << "GFLOPS (per second): " << gflops << std::endl;

    return 0;
}