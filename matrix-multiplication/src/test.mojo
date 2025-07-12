from benchmark import run, keep, clobber_memory, Report
from testing import assert_almost_equal
from algorithm import vectorize
from time import perf_counter_ns
from sys.info import is_x86, has_sse4, has_avx, has_avx2, has_avx512f, has_vnni, is_apple_silicon, is_apple_m1, is_apple_m2, is_apple_m3, has_neon, has_neon_int8_dotprod, has_neon_int8_matmul, num_physical_cores, num_logical_cores, num_performance_cores, simdbitwidth, os_is_macos, os_is_linux, os_is_windows, is_little_endian, is_64bit, CompilationTarget

alias Type = DType.float32
alias MatmulSignature = fn[Type: DType](Matrix[Type], Matrix[Type]) -> Matrix[Type]

fn basic_matmul[Type: DType](a: Matrix[Type], b: Matrix[Type]) -> Matrix[Type]:
    n = a.rows
    m = b.cols
    h = a.cols

    var res = Matrix[Type](n, m)

    for i in range(n):
        for j in range(m):
            for k in range(h):
                res[i, j] += a[i, k] * b[k, j]

    return res^


alias SCENARIOS = InlineArray[size=11](
    InlineArray[size=3](1, 1, 1), 
    InlineArray[size=3](1, 47, 97), 
    InlineArray[size=3](53, 1, 101), 
    InlineArray[size=3](17, 59, 103), 
    InlineArray[size=3](128, 128, 128), 
    InlineArray[size=3](128, 3072, 768), 
    InlineArray[size=3](512, 512, 512), 
    InlineArray[size=3](256, 1024, 4096), 
    InlineArray[size=3](1024, 1024, 1024),
    InlineArray[size=3](1300, 1024, 1024),  
    InlineArray[size=3](4096, 4096, 8192)
)
alias TYPES = InlineArray[size=7](DType.int8, DType.int16, DType.int32, DType.int64, DType.float16, DType.float32, DType.float64)

fn print_system_specs():
    print("System Specs", end=" | ")
    print("CPU: ", end="")
    if CompilationTarget.is_x86():
        print("x86", end=" ")
        if CompilationTarget.has_sse4(): print("SSE4", end=" ")
        if has_avx(): print("AVX", end=" ")
        if has_avx2(): print("AVX2", end=" ")
        if has_avx512f(): print("AVX512", end=" ")
        if has_vnni(): print("VNNI", end=" ")
        print("", end="| ")
    elif is_apple_silicon():
        print("Apple Silicon", end=" ")
        if is_apple_m1(): print("M1", end=" ")
        elif is_apple_m2(): print("M2", end=" ")
        elif is_apple_m3(): print("M3", end=" ")
        print("", end="| ")
    elif has_neon():
        print("ARM Neon", end=" ")
        if has_neon_int8_dotprod(): print("DotProd", end=" ")
        if has_neon_int8_matmul(): print("I8MM", end=" ")
        print("", end=" | ")
    print("Cores: Physical =", num_physical_cores(), "- Logical =", num_logical_cores(), "- Performance =", num_performance_cores(), end=" | ")
    print("SIMD width:", simdbitwidth(), "bits", end=" | ")
    print("OS: ", end=" ")
    if os_is_macos(): print("macOS", end=" | ")
    elif os_is_linux(): print("Linux", end=" | ")
    elif os_is_windows(): print("Windows", end=" | ")
    else: print("Unknown", end=" | ")
    print("Endianness:", "Little" if is_little_endian() else "Big", end=" | ")
    print("Bit width:", "64-bit" if is_64bit() else "32-bit")


fn test_matmul[matmul: MatmulSignature]() raises:
    @parameter
    for i in range(len(SCENARIOS) // 2):
        alias SCENARIO = SCENARIOS[i]
        var a = Matrix[Type].rand(SCENARIO[0], SCENARIO[2])
        var b = Matrix[Type].rand(SCENARIO[2], SCENARIO[1])
        var res = matmul(a, b)
        var correct = basic_matmul(a, b)

        for i in range(SCENARIO[0] * SCENARIO[1]): 
            assert_almost_equal(res.data[i], correct.data[i], atol=1e-5)
        print("✅ Passed test with M =", SCENARIO[0], ", N =", SCENARIO[1], ", K =", SCENARIO[2])


fn bench_matmul[matmul: MatmulSignature]() raises:
    print_system_specs()

    print("\nInformation in FLOPS/ns")

    print("M, N, K", end=" | ")
    for j in range(len(SCENARIOS)):
        print(SCENARIOS[j][0], SCENARIOS[j][1], SCENARIOS[j][2], end=" | ")
    print("Average |")

    @parameter
    for i in range(len(TYPES)):
        alias Type = TYPES[i]
        var total: Float64 = 0
        print(Type, end="")
        for _ in range(7 - len(String(Type))): print(" ", end="")
        print(" | ", end="")
        @parameter
        for j in range(len(SCENARIOS)):
            alias Dims = SCENARIOS[j]
            var res = Matrix[Type](Dims[0], Dims[1])
            var a = Matrix[Type].rand(Dims[0], Dims[2])
            var b = Matrix[Type].rand(Dims[2], Dims[1])
            fn wrapped_matmul() capturing: res = matmul(a, b)
            clobber_memory()
            var report = run[wrapped_matmul]()
            keep(res.data)
            keep(a.data)
            keep(b.data)
            var flops = Float64(Dims[0] * Dims[1] * Dims[2] * 2) / report.mean(unit="ns")
            total += flops
            print(String(flops)[0:7], end="")
            for _ in range(len(String(Dims[0])) + len(String(Dims[1])) + len(String(Dims[2])) + 2 - len(String(flops)[0:7])):
                print(" ", end="")
            print(" | ", end="")
        print(String(total / len(SCENARIOS))[0:7], end=" |\n")
