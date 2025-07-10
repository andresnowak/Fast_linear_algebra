from sys import info
from algorithm.functional import vectorize
from memory import stack_allocation, memset_zero, memset
from math import fma
from testing import assert_almost_equal

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
    if Type == DType.float64 or Type == DType.int64:
        return (10, 4)
    if Type == DType.float32 or Type == DType.int32:
        return (16, 6)
    if Type == DType.float16 or Type == DType.int16 or Type == DType.bfloat16:
        return (16, 12)
    if Type == DType.int8:
        return (16, 28)

    return (0, 0)


fn copy_pad_blockA[Type: DType](a: Matrix[Type]):
    pass


@always_inline
fn micro_kernel[
    Type: DType, //, mR: Int, nR: Int
](
    mut res: Matrix[Type],
    a: Matrix[Type],
    b: Matrix[Type],
    nr: Int,
    mr: Int,
    k: Int,
):
    # For us we say nR is from matrix a and mR is from matrix b

    alias NELTS = info.simdwidthof[Type]()

    var c_accumulator = stack_allocation[nR * mR * NELTS, Type]()
    memset_zero[count = nR * mR * NELTS](c_accumulator)

    var a_broadcasted_register = SIMD[Type, NELTS]()

    for p in range(k):
        for i in range(nR):

            @parameter
            fn vectorize_j[nelts: Int](j: Int):
                if nelts == NELTS:
                    a_broadcasted_register = a[nr + i, p]

                    c_accumulator.store[width=NELTS](
                        i * mR + j,
                        c_accumulator.load[width=NELTS](i * mR + j)
                        + b.load[NELTS](p, mr + j) * a_broadcasted_register,
                    )
                else:
                    var a_broadcasted_register = SIMD[Type, nelts](a[nr + i, p])

                    c_accumulator.store[width=nelts](
                        i * mR + j,
                        c_accumulator.load[width=nelts](i * mR + j)
                        + b.load[nelts](p, mr + j) * a_broadcasted_register,
                    )

            vectorize[vectorize_j, NELTS, unroll_factor=mR](mR)

    @parameter
    for i in range(nR):
        @parameter
        fn vectorize_j_store[nelts: Int](j: Int):
        # for j in range(mR):
            var res_pos = (nr + i) * res.cols + mr + j
            res.store[width=nelts](nr + i, mr + j, c_accumulator.load[width=nelts](i * mR + j))
            # res[nr + i, mr + j] = c_accumulator[i * mR + j]
        
        vectorize[vectorize_j_store, NELTS, unroll_factor=mR](mR)


fn matmul[Type: DType](a: Matrix[Type], b: Matrix[Type]) -> Matrix[Type]:
    if a.cols != b.rows:
        print("A cols and B rows have to be equal")
        return Matrix[Type](0, 0)

    n = a.rows
    m = b.cols
    k = a.cols

    var res = Matrix[Type](n, m)

    alias mR_nR = get_nr_mr[Type]()
    # here we say a is size (n, k) and b is size (k, m), so mR belongs to b
    alias mR = mR_nR[0]
    alias nR = mR_nR[1]

    for mr in range(0, m, mR):
        for nr in range(0, n, nR):
            micro_kernel[mR, nR](res, a, b, nr, mr, k)
    return res^


alias MatmulSignature = fn[Type: DType] (Matrix[Type], Matrix[Type]) -> Matrix[
    Type
]


fn test_matmul[matmul: MatmulSignature]() raises:
    var a = Matrix[DType.float32].rand(6 * 4, 24)
    var b = Matrix[DType.float32].rand(24, 16 * 4)
    var res = matmul(a, b)
    var correct = basic_matmul(a, b)

    for i in range(6 * 4 * 16 * 4):
        assert_almost_equal(res.data[i], correct.data[i], atol=1e-5)
    print("✅ Passed test with M =", 6 * 4, ", N =", 16 * 4, ", K =", 24)


fn main() raises:
    a = Matrix[DType.float32].randint(6 * 4, 24)
    b = Matrix[DType.float32].randint(24, 16 * 4)

    print(a)
    print()
    print(b)
    print("\n\n")
    print(matmul(a, b))

    test_matmul[matmul]()
