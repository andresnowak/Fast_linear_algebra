from src.test import test_matmul, bench_matmul
from gemm_naive import matmul as matmul_naive
from gemm_reorder import matmul as matmul_reorder
from gemm_simd import matmul as matmul_simd
from gemm_micro_kernel import matmul as matmul_micro_kernel

fn main() raises:
    test_matmul[matmul_simd]()
    bench_matmul[matmul_simd]()