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


fn get_mr_nr[Type: DType]() -> Tuple[Int, Int]:
    # mr is always a multiple of simdwidth
    if Type == DType.float64 or Type == DType.int64:
        return (10, 4)
    if Type == DType.float32 or Type == DType.int32:
        return (16, 6)
    if Type == DType.float16 or Type == DType.int16 or Type == DType.bfloat16:
        return (16, 12)
    if Type == DType.int8:
        return (16, 26)  # return (16, 28)

    return (0, 0)


fn get_nc_mc_kc[Type: DType]() -> Tuple[Int, Int, Int]:
    # mr is always a multiple of simdwidth
    if Type == DType.float64 or Type == DType.int64:
        return (1000, 1000, 1000)
    if Type == DType.float32 or Type == DType.int32:
        return (1020, 1024, 1000)
    if Type == DType.float16 or Type == DType.int16 or Type == DType.bfloat16:
        return (1200, 1024, 1000)
    if Type == DType.int8:
        return (1300, 1024, 1000)  # return (16, 28)

    return (0, 0, 0)


fn blockA_panel[
    Type: DType, //, nC: Int, kC: Int, nR: Int
](
    mut blockA_buffer: Matrix[Type],
    a: Matrix[Type],
    ic: Int,
    pc: Int,
    ir: Int,
    nr: Int,
    kc: Int,
):
    var panel_number = ir * kc # because we only fill kc values and because we are filling the data in an iterative way we have to move ir * kc (meaning we move (ir / nR) panels) to go the present panel and then we just fill the data one by one. (Our panels are of size nR * kc in the a buffer just to not fill the kC values as it is not necessary)
    var panel_position = 0

    for p in range(kc):
        for i in range(nr): # iterate the panel by rows so as to convert this rows to connect them in a row major way
            blockA_buffer.data[ir * kc + panel_position] = a[
                i + ir + ic, pc + p
            ]
            panel_position += 1

        for i in range(nr, nR):
            blockA_buffer.data[ir * kc + panel_position] = 0
            panel_position += 1


fn blockA_packed[
    Type: DType, //, nC: Int, kC: Int, nR: Int
](
    mut blockA_buffer: Matrix[Type],
    a: Matrix[Type],
    ic: Int,
    pc: Int,
    nc: Int,
    kc: Int,
):
    # creating our panel of size nR * kC
    for ir in range(0, nc, nR):  # The panel we are on
        var nr = min(nR, nc - ir)
        blockA_panel[nC, kC, nR](blockA_buffer, a, ic, pc, ir, nr, kc)


fn blockB_panel[
    Type: DType, //, mC: Int, kC: Int, mR: Int
](
    mut blockB_buffer: Matrix[Type],
    b: Matrix[Type],
    jc: Int,
    pc: Int,
    jr: Int,
    mr: Int,
    kc: Int,
):
    var panel_number = jr * kc # because we only fill kc values and because we are filling the data in an iterative way we have to move jr * kc (meaning we move (jr / mR) panels) to go the present panel and then we just fill the data one by one. (Our panels are of size nR * kc in the a buffer just to not fill the kC values as it is not necessary)
    var panel_position = 0

    for p in range(kc):
        for j in range(mr):
            blockB_buffer.data[panel_number + panel_position] = b[
                p + pc, j + jc + jr
            ]
            panel_position += 1
        for j in range(mr, mR):
            blockB_buffer.data[panel_number + panel_position] = 0
            panel_position += 1


fn blockB_packed[
    Type: DType, //, mC: Int, kC: Int, mR: Int
](
    mut blockB_buffer: Matrix[Type],
    b: Matrix[Type],
    jc: Int,
    pc: Int,
    mc: Int,
    kc: Int,
):
    # creating our panel of size kC * mR
    for jr in range(0, mc, mR):  # The panel we are on
        var mr = min(mR, mc - jr)
        blockB_panel[mC, kC, mR](blockB_buffer, b, jc, pc, jr, mr, kc)


@always_inline
fn micro_kernel[
    Type: DType, //, mR: Int, nR: Int
](
    mut res: Matrix[Type],
    a: Matrix[Type],
    b: Matrix[Type],
    ir: Int,
    jr: Int,
    kc: Int,
    nr: Int,
    mr: Int,
):
    # For us we say nR is from matrix A and mR is from matrix B
    # 1 register for broadcasted value of A, mR/NELTS registers for B and mR/NELTS * nR registers for C_accumulator

    alias NELTS = info.simdwidthof[Type]()

    alias alignment = alignof[SIMD[Type, NELTS]]()
    var c_accumulator = stack_allocation[nR * mR, Type, alignment=alignment]()
    memset_zero[count = nR * mR](c_accumulator)

    var a_vecs = InlineArray[SIMD[Type, NELTS], nR](uninitialized=True)


    count_b_position = 0
    count_a_position = 0

    for p in range(kc):

        @parameter
        for i in range(nR):
            panel_a_number = ir * kc
            a_vecs[i] = a.data[panel_a_number + count_a_position + i]

        count_a_position += nR

        @parameter
        for i in range(nR):

            @parameter
            for j in range(0, mR, NELTS):
                panel_b_number = jr * kc

                c_accumulator.store[width=NELTS](
                    i * mR + j,
                    fma(
                        a_vecs[i],
                        b.data.load[width=NELTS](
                            panel_b_number + (count_b_position + j)
                        ),
                        c_accumulator.load[width=NELTS](
                            i * mR + j
                        ),
                    ),
                )
            
        count_b_position += mR

    if mr != mR:
        for i in range(nr):

            @parameter
            fn vectorize_j_store[nelts: Int](j: Int):
                res.store[width=nelts](
                    ir + i, jr + j, c_accumulator.load[width=nelts](i * mR + j)
                )

            vectorize[vectorize_j_store, NELTS](mr)
    else:
        for i in range(nr):

            @parameter
            for j in range(0, mR, NELTS):
                res.store[width=NELTS](
                    ir + i, jr + j, c_accumulator.load[width=NELTS](i * mR + j)
                )


fn matmul[Type: DType](a: Matrix[Type], b: Matrix[Type]) -> Matrix[Type]:
    if a.cols != b.rows:
        print("A cols and B rows have to be equal")
        return Matrix[Type](0, 0)

    var N = a.rows
    var M = b.cols
    var K = a.cols

    var res = Matrix[Type](N, M)

    alias mR_nR = get_mr_nr[Type]()
    # here we say a is size (n, k) and b is size (k, m), so mR belongs to b
    alias mR = mR_nR[0]
    alias nR = mR_nR[1]

    alias nC_mC_kC = get_nc_mc_kc[Type]()
    alias nC = nC_mC_kC[0]
    alias mC = nC_mC_kC[1]
    alias kC = nC_mC_kC[2]

    var blockA_buffer = Matrix[Type](nC, kC)
    var blockB_buffer = Matrix[Type](kC, mC)

    for ic in range(0, N, nC):
        var nc = min(nC, N - ic)
        for pc in range(0, K, kC):
            var kc = min(kC, K - pc)

            blockA_packed[nC, kC, nR](blockA_buffer, a, ic, pc, nc, kc)
            for jc in range(0, M, mC):
                var mc = min(mC, M - jc)

                blockB_packed[mC, kC, mR](blockB_buffer, b, jc, pc, mc, kc)
                for ir in range(0, nc, nR):
                    var nr = min(nR, nc - ir)

                    for jr in range(0, mc, mR):
                        var mr = min(mR, mc - jr)

                        micro_kernel[mR, nR](
                            res,
                            blockA_buffer,
                            blockB_buffer,
                            ir,
                            jr,
                            kc,
                            nr,
                            mr,
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
    a = Matrix[DType.int8].randint(1300, 1024)
    b = Matrix[DType.int8].randint(1024, 1024)

    # print(a)
    # print()
    # print(b)
    # print("\n\n")
    # print(matmul(a, b))

    test_matmul[matmul]()

    matmul(a, b)
