from src.test import test_matmul, bench_matmul
from gemm_naive import matmul as matmul_naive
from gemm_reorder import matmul as matmul_reorder

fn main() raises:
    test_matmul[matmul_reorder]()
    bench_matmul[matmul_reorder]()