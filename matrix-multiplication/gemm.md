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