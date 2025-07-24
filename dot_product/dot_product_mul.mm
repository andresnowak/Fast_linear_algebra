#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

#include "dot_product.h"
#include "test.h"

std::pair<float, double> dot_product_mul(const std::vector<float> &a, const std::vector<float> &b) {
    size_t n = a.size();

    auto reduce = [](size_t n, float* results) -> float {
        float dot = 0;
    
        for (size_t i = 0; i < n; ++i) {
            dot += results[i];
        }

        return dot;
    };

    int gridSize[3] = {static_cast<int>(n), 1, 1};
    int threadGroupSize[3] = {256, 1, 1};
    return test_dot_product(a, b, reduce, gridSize, threadGroupSize);
}
