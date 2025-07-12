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

## GEMM Micro-Kernel

Now after seeing about the loop reordering and SIMD operations, let’s talk about what is the most important part of matrix multiplication: the **micro-kernel** (or accumulator, etc...). The idea here is that we work on a block of size $(mR \times nR)$, which we’ll call $\bar{C}$, and instead of working directly with the full $A$ and $B$ matrices, we work on smaller submatrices $\bar{A}$ ($nR \times K$) and $\bar{B}$ ($mR \times K$).

The purpose of this micro-kernel is twofold:
1. **Reduce the number of memory accesses**.
2. **Maximize the use of registers**, which allows us to access data faster.

Let’s first look at the original, normal operation of multiplying rows of $\bar{A}$ with columns of $\bar{B}$. In this case, we end up with $2K \times mR \times nR$ memory accesses. Why $2K$? Because for every $K$, we access both the rows of $\bar{A}$ and the columns of $\bar{B}$.

But now let’s consider the idea of doing **outer products**. So Instead of accessing rows of $\bar{A}$ and columns of $\bar{B}$ and doing our multiplication of both and sum the values, we now access **columns of $\bar{A}$** and **rows of $\bar{B}$**. For each value in a column of $\bar{A}$, we broadcast it and compute an outer product with the row of $\bar{B}$. This gives us the first values for the first row of $\bar{C}$. Then we move to the next value in the column of $\bar{A}$, do the same, and get the first values of the next row of $\bar{C}$. After repeating this $K$ times for all rows and columns, we end up with only $(mR + nR) \times K$ memory accesses. This difference compared to $2K \times mR \times nR$ becomes more noticeable as $(mR + nR)K$ grows larger (by having bigger $mR$ and $nR$ values).


### Determining $mR$ and $nR$

To figure out the sizes of $mR$ and $nR$, we need to think about how many SIMD registers (e.g., Vector registers) we have. For example, on an M1 processor, we have 32 Vector registers. Based on what we said earlier, we need:
- $mR \times nR$ registers for $\bar{C}$,
- $mR$ registers for rows of $\bar{B}$,
- 1 register for broadcasting values from columns of $\bar{A}$ (since we can reuse the same register for each broadcast).

So, the total number of registers is constrained by:
$$
\left(\frac{mR}{\text{simd\_width}} \times nR + \frac{mR}{\text{simd\_width}} + 1\right) \leq 32
$$

To minimize memory accesses, $mR$ and $nR$ should ideally be the same. Why? Because if $mR \times nR = P$ (a constant), then the sum $mR + nR = S$ is smallest when $mR = nR$. To see this, let’s set $nR = \frac{P}{mR}$. Then:
$$
mR + \frac{P}{mR} = S
$$
The smallest value of $S$ occurs when the derivative is zero:
$$
1 - \frac{P}{mR^2} = 0 \quad \Rightarrow \quad mR = \sqrt{P}, \quad nR = \frac{P}{\sqrt{P}} = \sqrt{P}.
$$
However, in practice, $mR$ must also be divisible by the SIMD width, and different values of $mR$ and $nR$ can sometimes yield better results depending on the hardware.


### Micro-Kernel Implementation

Now let’s look at the implementation of the micro-kernel. This is the core function that performs the computation for a block of size $(mR \times nR)$:

```python
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
    # For us we say nR is from matrix A and mR is from matrix B
    # 1 register for broadcasted value of A, mR/NELTS registers for B and mR/NELTS * nR registers for C_accumulator

    alias NELTS = info.simdwidthof[Type]()

    alias alignment = alignof[SIMD[Type, NELTS]]()
    var c_accumulator = stack_allocation[nR * mR, Type, alignment=alignment]()
    memset_zero[count = nR * mR](c_accumulator)

    var a_vecs = InlineArray[SIMD[Type, NELTS], nR](uninitialized=True)

    for p in range(K):
        
        @parameter
        for i in range(nR):
            a_vecs[i] = a[nr_a + i, p]
            # a_broadcasted_register = a[nr_a + i, p]
        
        @parameter
        for i in range(nR):
            @parameter
            for j in range(0, mR, NELTS):
                c_accumulator.store[width=NELTS](
                    i * mR + j,
                    fma(a_vecs[i], b.load[NELTS](p, mr_b + j), c_accumulator.load[width=NELTS](i * mR + j))
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
```

And after our accumulator is computed we copy the values of the accumulator back to the $C$ matrix and we continue with our next block (or tile) until we compute the whole $C$ result.


### Handling Edge Cases

One thing to consider is that $M$ and $N$ are not always divisible by $mR$ and $nR$. When we’re at the edges of the matrices, our $(mR \times nR)$ block might go out of bounds. To handle this, we can pad the $\bar{A}$ and $\bar{B}$ values with zeros, allowing us to still perform unrolled $mR$ and $nR$ operations.


### Matrix Multiplication Implementation

Here’s the full matrix multiplication implementation using the micro-kernel:

```python
fn matmul[Type: DType](a: Matrix[Type], b: Matrix[Type]) -> Matrix[Type]:
    if a.cols != b.rows:
        print("A cols and B rows have to be equal")
        return Matrix[Type](0, 0)

    var N = a.rows
    var M = b.cols
    var K = a.cols

    var res = Matrix[Type](N, M)

    alias mR_nR = get_nr_mr[Type]()
    # a is size (n, k), b is size (k, m), so mR belongs to b
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

    return res
```