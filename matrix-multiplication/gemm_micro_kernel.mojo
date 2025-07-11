from sys import info, alignof, prefetch
from algorithm.functional import vectorize
from memory import stack_allocation, memset_zero, memset, UnsafePointer
from math import fma
from testing import assert_almost_equal
from sys.intrinsics import masked_store

from src.matrix import Matrix
from src.test import basic_matmul

# This is for m1 processor


fn get_nelts[Type: DType]() -> Int:
    # simdwidthof(T) = #lanes per vector * #vector‐ops you can issue per cycle
    if info.is_apple_silicon():
        return 4 * info.simdwidthof[Type]()
    else:
        return 2 * info.simdwidthof[Type]()


fn get_nr_mr[Type: DType]() -> Tuple[Int, Int]:
    # mr is always a multiple of simdwidth
    if Type == DType.float64 or Type == DType.int64:
        return (10, 4)
    if Type == DType.float32 or Type == DType.int32:
        return (16, 6)
    if Type == DType.float16 or Type == DType.int16 or Type == DType.bfloat16:
        return (16, 12)
    if Type == DType.int8:
        return (16, 28)

    return (0, 0)


fn copy_pad_blockA[
    Type: DType, //, nR: Int
](mut blockA_buffer: Matrix[Type], a: Matrix[Type], nr: Int, n: Int, K: Int):
    alias NELTS = info.simdwidthof[Type]()

    for i in range(n):

        @parameter
        fn vectorize_k[nelts: Int](p: Int):
            blockA_buffer.store[nelts](i, p, a.load[nelts](nr + i, p))

        vectorize[vectorize_k, NELTS](K)

    @parameter
    fn vectorize_pad[nelts: Int](i: Int):
        blockA_buffer.data.store[width=nelts](n * K + i, 0)

    vectorize[vectorize_pad, NELTS]((nR - n) * K)


fn copy_pad_blockB[
    Type: DType, //, mR: Int
](mut blockB_buffer: Matrix[Type], b: Matrix[Type], mr: Int, m: Int, K: Int):
    alias NELTS = info.simdwidthof[Type]()

    for p in range(K):
        @parameter
        fn vectorize_copy[nelts: Int](j: Int):
            blockB_buffer.store[nelts](
                p, j,
                b.load[nelts](p, mr + j)
            )
        vectorize[vectorize_copy, NELTS](m)

        @parameter
        fn vectorize_pad[nelts: Int](j: Int):
            blockB_buffer.store[nelts](p, m + j, 0)
        vectorize[vectorize_pad, NELTS](mR - m)


@always_inline
fn micro_kernel[
    Type: DType, //, mR: Int, nR: Int
](
    mut res: Matrix[Type],
    a: Matrix[Type],
    b: Matrix[Type],
    nr: Int,
    mr: Int,
    nr_a: Int,
    mr_b: Int,
    K: Int,
    n: Int,
    m: Int,
):
    # For us we say nR is from matrix a and mR is from matrix b

    alias NELTS = info.simdwidthof[Type]()

    alias alignment = alignof[SIMD[Type, NELTS]]()
    var c_accumulator = stack_allocation[nR * mR, Type, alignment=alignment]()
    memset_zero[count = nR * mR](c_accumulator)

    for p in range(K):

        @parameter
        for i in range(nR):
            var a_broadcasted_register = SIMD[Type, NELTS](a[nr_a + i, p])

            @parameter
            for j in range(0, mR, NELTS):
                c_accumulator.store[width=NELTS](
                    i * mR + j,
                    fma(a_broadcasted_register, b.load[NELTS](p, mr_b + j), c_accumulator.load[width=NELTS](i * mR + j))
                )

    if m != mR:
        for i in range(n):

            @parameter
            fn vectorize_j_store[nelts: Int](j: Int):
                var res_pos = (nr + i) * res.cols + mr + j
                res.store[width=nelts](
                    nr + i, mr + j, c_accumulator.load[width=nelts](i * mR + j)
                )

            vectorize[vectorize_j_store, NELTS](m)
    else:
        for i in range(n):

            @parameter
            for j in range(0, mR, NELTS):
                var res_pos = (nr + i) * res.cols + mr + j
                res.store[width=NELTS](
                    nr + i, mr + j, c_accumulator.load[width=NELTS](i * mR + j)
                )


fn matmul[Type: DType](a: Matrix[Type], b: Matrix[Type]) -> Matrix[Type]:
    if a.cols != b.rows:
        print("A cols and B rows have to be equal")
        return Matrix[Type](0, 0)

    var N = a.rows
    var M = b.cols
    var K = a.cols

    var res = Matrix[Type](N, M)

    alias mR_nR = get_nr_mr[Type]()
    # here we say a is size (n, k) and b is size (k, m), so mR belongs to b
    alias mR = mR_nR[0]
    alias nR = mR_nR[1]

    var blockA_buffer = Matrix[Type](nR, K)
    var blockB_buffer = Matrix[Type](K, mR)

    for mr in range(0, M, mR):
        var m = min(mR, M - mr)
        var blockB = UnsafePointer(to=b)

        var mr_blockB = mr
        if m != mR:
            copy_pad_blockB[mR](blockB_buffer, b, mr, m, K)
            blockB = UnsafePointer(to=blockB_buffer)
            mr_blockB = 0

        for nr in range(0, N, nR):
            var n = min(nR, N - nr)
            var blockA = UnsafePointer(to=a)

            var nr_blockA = nr
            if n != nR:
                copy_pad_blockA[nR](blockA_buffer, a, nr, n, K)
                blockA = UnsafePointer(to=blockA_buffer)
                nr_blockA = 0

            micro_kernel[mR, nR](
                res, blockA[], blockB[], nr, mr, nr_blockA, mr_blockB, K, n, m
            )

    return res^


alias MatmulSignature = fn[Type: DType] (Matrix[Type], Matrix[Type]) -> Matrix[
    Type
]


fn test_matmul[matmul: MatmulSignature]() raises:
    var N = 51
    var M = 16 * 4 + 5
    var K = 24

    var a = Matrix[DType.float32].rand(N, K)
    var b = Matrix[DType.float32].rand(K, M)
    var res = matmul(a, b)
    var correct = basic_matmul(a, b)

    for i in range(N * M):
        assert_almost_equal(res.data[i], correct.data[i], atol=1e-5)
    print("✅ Passed test with M =", M, ", N =", N, ", K =", K)


fn main() raises:
    a = Matrix[DType.float32].randint(6 * 4, 24)
    b = Matrix[DType.float32].randint(24, 16 * 4)

    print(a)
    print()
    print(b)
    print("\n\n")
    print(matmul(a, b))

    test_matmul[matmul]()
