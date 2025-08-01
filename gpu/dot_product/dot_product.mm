#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <omp.h>

#include "dot_product.h"
#include "test.h"

std::pair<float, double> dot_product_mul(const std::vector<float> &a, const std::vector<float> &b) {
    size_t n = a.size();

    auto reduce = [](size_t n, float* results) -> float {
        float dot = 0;

        #pragma omp parallel for reduction(+:dot)
        for (size_t i = 0; i < n; ++i) {
            dot += results[i];
        }

        return dot;
    };

    int gridSize[3] = {(static_cast<int>(n) + 1024 - 1) / 1024 * 1024, 1, 1};
    int threadGroupSize[3] = {1024, 1, 1};
    return test_dot_product(a, b, reduce, gridSize, threadGroupSize, @"dot_product_mul.metallib");
}

std::pair<float, double> dot_product_mul_reduce(const std::vector<float> &a, const std::vector<float> &b) {
    size_t n = a.size();

    int gridSize[3] = {(static_cast<int>(n) + 1024 - 1) / 1024 * 1024, 1, 1};
    int threadGroupSize[3] = {1024, 1, 1};

    auto reduce = [](size_t n, float* results) -> float {
        float dot = 0;

        // #pragma omp parallel for reduction(+:dot) schedule(dynamic, 10) // change the chunk sizes
        #pragma omp parallel for reduction(+:dot)
        for (size_t i = 0; i < ceil(n / 1024.0); ++i) {
            dot += results[i];
        }

        return dot;
    };

    return test_dot_product(a, b, reduce, gridSize, threadGroupSize, @"dot_product_mul_reduce.metallib");
}

std::pair<float, double> dot_product_mul_reduce_atomic(const std::vector<float> &a, const std::vector<float> &b) {
    size_t n = a.size();

    int gridSize[3] = {(static_cast<int>(n) + 1024 - 1) / 1024 * 1024, 1, 1};
    int threadGroupSize[3] = {1024, 1, 1};

    auto reduce = [](size_t n, float* results) -> float {
        return results[0];
    };

    return test_dot_product(a, b, reduce, gridSize, threadGroupSize, @"dot_product_mul_reduce_atomic.metallib");
}

std::pair<float, double> dot_product_mul_tree_reduce(const std::vector<float> &a, const std::vector<float> &b) {
    size_t n = a.size();

    int gridSize[3] = {(static_cast<int>(n) + 1024 - 1) / 1024 * 1024, 1, 1}; // we know n is 2048
    int threadGroupSize[3] = {1024, 1, 1};

    auto reduce = [](size_t n, float* results) -> float {
        return results[0];
    };

    return test_dot_product(a, b, reduce, gridSize, threadGroupSize, @"dot_product_mul_tree_reduce.metallib");
}

std::pair<float, double> dot_product_mul_tree_reduce_fix_divergence(const std::vector<float> &a, const std::vector<float> &b) {
    size_t n = a.size();

    int gridSize[3] = {(static_cast<int>(n) + 1024 - 1) / 1024 * 1024, 1, 1}; // we know n is 2048
    int threadGroupSize[3] = {1024, 1, 1};

    auto reduce = [](size_t n, float* results) -> float {
        return results[0];
    };

    return test_dot_product(a, b, reduce, gridSize, threadGroupSize, @"dot_product_mul_tree_reduce_fix_divergence.metallib");
}

std::pair<float, double> dot_product_mul_reduce_hierarchical_reduction(const std::vector<float> &a, const std::vector<float> &b) {
    size_t n = a.size();

    int gridSize[3] = {(static_cast<int>(n) + 1024 - 1) / 1024 * 1024, 1, 1};
    int threadGroupSize[3] = {1024, 1, 1};

    auto reduce = [](size_t n, float* results) -> float {
        return results[0];
    };

    return test_dot_product(a, b, reduce, gridSize, threadGroupSize, @"dot_product_mul_reduce_hierarchical_reduction.metallib");
}