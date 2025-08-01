from src.matrix import Matrix

fn matmul[Type: DType](a: Matrix[Type], b: Matrix[Type]) -> Matrix[Type]:
    if a.cols != b.rows:
        print("A cols and B rows have to be equal")
        return Matrix[Type](0, 0)

    n = a.rows
    m = b.cols
    h = a.cols

    var res = Matrix[Type](n, m)

    for i in range(n):
        for j in range(m):
            for k in range(h):
                res[i, j] += a[i, k] * b[k, j]

    return res^

    

fn main() raises:
    a = Matrix[DType.float32].randint(4, 3)
    b = Matrix[DType.float32].randint(3, 2)

    print(a)
    print()
    print(b)
    print()
    print(matmul(a, b))