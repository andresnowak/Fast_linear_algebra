#pragma once

#include <vector>

std::pair<float, double> dot_product_mul(const std::vector<float> &A, const std::vector<float> &B);

std::pair<float, double> dot_product_mul_reduce(const std::vector<float> &A, const std::vector<float> &B);

std::pair<float, double> dot_product_mul_reduce_atomic(const std::vector<float> &A, const std::vector<float> &B);

std::pair<float, double> dot_product_mul_tree_reduce(const std::vector<float> &A, const std::vector<float> &B);