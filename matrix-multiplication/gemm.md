# GEMM

## GEMM Naive

The naive GEMM implementation follows the mathematical definition of matrix multiplication using three nested loops. Here's how it works:
Core Algorithm Logic:

- Outer loop (i): Iterates through each row of matrix A
- Middle loop (j): Iterates through each column of matrix B
- Inner loop (k): Computes the dot product between row i of A and column j of B

Step-by-step execution:

- For position (i,j) in the result matrix, we initialize with zero
- We multiply corresponding elements: A[i,k] × B[k,j] for each k
- We accumulate these products to get the final value for result[i,j]

Memory access pattern:

- So for Matrix A we are accessing row-wise, we have good cache locality here because our data is stored in Row-Major
- Matrix B is accessed column-wise, here we have poor cache locality.
- Result matrix is accessed in row-major order, because each time we move a column on Matrix B to compute next Res[i, j] we are moving column wise in Res matrix

```python
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
```

## GEMM Reorder

To make the cache locality or Matrix B better we reorder the loops

- Outer loop (i): Iterates through each row of matrix A
- Middle loop (k):  Computes the dot product between row i of A and column j of B
- Inner loop (j): Iterates through each column of matrix B

Now the memory access pattern for Matrix B is that now we are also accessing the data in a Row wise manner so we are now maintaining good cache locality for Matrix B

Step-by-step execution:

- For position (i,j) in the result matrix, we initialize with zero
- We multiply corresponding elements: A[i,k] × B[k,j], where first for i=0 and k=0 we multiply A[i, k] by all the columns of Matrix B[k, j] in row k
- We then iterate by k and continue as normal
- And in the end We accumulate these products to get the final value for result[i,j]

```python
fn matmul[Type: DType](a: Matrix[Type], b: Matrix[Type]) -> Matrix[Type]:
if a.cols != b.rows:
    print("A cols and B rows have to be equal")
    return Matrix[Type](0, 0)

n = a.rows
m = b.cols
h = a.cols

var res = Matrix[Type](n, m)

for i in range(n):
    for k in range(h):
        for j in range(m):
                res[i, j] += a[i, k] * b[k, j]

return res^
```

## GEMM SIMD

Now after this we can apply data parallelization (SIMD) to accelerate our computation of the operations

- Outer loop (i): Iterates through each row of matrix A
- Middle loop (k):  Computes the dot product between row i of A and column j of B
- Inner loop (j): Iterates through each column of matrix $B$ in a vectorized manner, operating on $N$ values (denoted as `nelts`) in parallel. For example, on an M1 processor with 128-bit SIMD, using `float32` allows us to process 4 values simultaneously (e.g., multiplying two $1 \times 4$ vectors).

Step-by-step execution:

- For position (i,j) in the result matrix, we initialize with zero
- We multiply corresponding elements: $A_{i,k} × [B_{k,j}, B_{k,j+1}, ..., B_{k,j+nelts-1}], where first for i=0 and k=0 we multiply A[i, k] by all the columns of Matrix B[k, j] in row k. The idea here is remember oen value of the row will be multiplied by all values in column of b, so a is broadcasted to work in parallel now in b
- We then iterate by k and continue as normal
- And in the end we accumulate these products to get the final value for [result_{k,j}, result_{k,j+1}, ..., result_{k,j+nelts-1}], so here the values are also stored in a data parallel way

Finally, we apply SIMD to handle larger sizes by leveraging the hardware's capabilities:  
- The number of operations performed in parallel depends on the **number of lanes per vector** and the **number of vector operations that can be issued per cycle**.  
- For example, on an M1 processor, in one cycle, we can perform multiple vector operations, allowing us to process more data in parallel.

```python
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
                res.store[width](i, j, res.load[width](i, j) + a[i, k] * b.load[width](k, j))
            
            vectorize[vectorize_j, NELTS](m)

    return res^
```