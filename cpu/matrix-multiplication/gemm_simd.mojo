from sys import info
from algorithm.functional import vectorize
from math import fma

from src.matrix import Matrix

fn get_nelts[Type: DType]() -> Int:
    # simdwidthof(T) = #lanes per vector * #vector‐ops you can issue per cycle
    if info.is_apple_silicon():
        return 4 * info.simdwidthof[Type]()
    else:
        return 2 * info.simdwidthof[Type]()

fn matmul[Type: DType](a: Matrix[Type], b: Matrix[Type]) -> Matrix[Type]:
    if a.cols != b.rows:
        print("A cols and B rows have to be equal")
        return Matrix[Type](0, 0)

    alias NELTS = get_nelts[Type]()

    n = a.rows
    m = b.cols
    h = a.cols

    var res = Matrix[Type](n, m)

    for i in range(n):
        for k in range(h):
            @parameter
            fn vectorize_j[width: Int](j: Int):
                res.store[width](i, j, fma(SIMD[Type, width](a[i, k]), b.load[width](k, j), res.load[width](i, j)))
            
            vectorize[vectorize_j, NELTS](m)

    return res^

    

fn main() raises:
    a = Matrix[DType.float32].randint(4, 3)
    b = Matrix[DType.float32].randint(3, 2)

    print(a)
    print()
    print(b)
    print()
    print(matmul(a, b))