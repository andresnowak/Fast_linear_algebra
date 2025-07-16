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


## GEMM Cache Blocking (BLIS-style 5-loop GEMM)

Now before parallelization there is a last important part that has to do with the memory hierarchy: **cache blocking**.  
Loading values from RAM is slow (~100 ns), while L3 cache is ~20 ns, L2 ~10 ns, and L1 ~1 ns.  
So if we can have a big chunk of the values that we will be reutilizing in our operations, why don’t we first load a part of our matrices into a copy that is of a smaller size (as caches are smaller than our RAM) and keep this data in faster memory for our operations?  
Remember that our data will be used in multiple operations: a **column (vertical)** of A will be used by the **whole row (horizontal)** of B, and since we are doing micro-kernels, in each micro-kernel we will be reutilising these rows and columns.

This is the **BLIS-style 5-loop GEMM**.

### The Blocking Idea

We divide our A and B matrices into cache blocks:

- $\bar{A}$ of size $n_C \times k_C$  
- $\bar{B}$ of size $k_C \times m_C$

These blocks are further divided into **panels**:

- $\bar{A}$ panels of size $n_R \times k_C$  
- $\bar{B}$ panels of size $k_C \times m_R$

Inside these panels we **reorder the data**:

- **Matrix A** is traversed **column-wise**, but stored **row-major**, so we are not utilising the cache lanes when loading our values.  
  → In our panels of $\bar{A}$ we put the data in a continuous way by putting our columns of size $n_R$ in a row-major way `[nR, nR, nR, ...]`.

- **Matrix B** is traversed **row-wise**, but stored **row-major**, so each time we iterate $m_R$ and go to the next row in the panel we have to jump positions in memory.  
  → Instead, we save our rows of size $m_R$ contiguously in our panels.

**Where do they live?**

- $\bar{B}$ lives in **L2 cache**  
- $\bar{A}$ lives in **L3 cache**  
- The **panels of A** (when using them) live in **L1 cache**

The idea is to **fill our caches as much as possible** so as to utilise them as much as we can.  
We want A panels on L1 cache because the values of A are the ones we **broadcast** when using the registers, so there are **fewer values in the registers** being used compared to B.  
Therefore, A needs L1 cache more than B panels; B lives mostly in the registers so it doesn't need L1 cache as much.

### Choosing the Block Sizes

```python
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
```

These numbers are tuned so that:

- $k_C$ keeps the working set of $\bar{B}$ under the L2 capacity.  
- $n_C$ and $m_C$ keep the working set of $\bar{A}$ under the L3 capacity.  
- $n_R$ and $m_R$ (from the micro-kernel) fit the L1 and register file.

### Packing the Panels

We **pack** the data into temporary buffers (`blockA_buffer`, `blockB_buffer`) before calling the micro-kernel.

#### Packing A

```python
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
    var panel_number = (
        ir * kc
    )  # because we only fill kc values and because we are filling the data in an iterative way we have to move ir * kc (meaning we move (ir / nR) panels) to go the present panel and then we just fill the data one by one. (Our panels are of size nR * kc in the a buffer just to not fill the kC values as it is not necessary)
    var panel_position = 0

    for p in range(kc):
        for i in range(
            nr
        ):  # iterate the panel by rows so as to convert this rows to connect them in a row major way
            blockA_buffer.data[ir * kc + panel_position] = a[
                i + ir + ic, pc + p
            ]
            panel_position += 1

        for i in range(nr, nR):
            blockA_buffer.data[ir * kc + panel_position] = 0
            panel_position += 1
```

#### Packing B

```python
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
    alias NELTS = info.simdwidthof[Type]()

    var panel_number = (
        jr * kc
    )  # because we only fill kc values and because we are filling the data in an iterative way we have to move jr * kc (meaning we move (jr / mR) panels) to go the present panel and then we just fill the data one by one. (Our panels are of size nR * kc in the a buffer just to not fill the kC values as it is not necessary)
    var panel_position = 0

    for p in range(kc):

        @parameter
        fn vectorize_j[nelts: Int](j: Int):
            blockB_buffer.data.store[width=nelts](
                panel_number + panel_position,
                b.load[nelts](p + pc, j + jc + jr),
            )
            panel_position += nelts

        vectorize[vectorize_j, NELTS](mr)

        @parameter
        fn copy_pad[nelts: Int](j: Int):
            # for j in range(mr, mR):
            blockB_buffer.data.store[width=nelts](
                panel_number + panel_position, 0
            )
            panel_position += nelts

        vectorize[copy_pad, NELTS](mR - mr)
```

### Putting It All Together

The **5-loop structure** (BLIS-style) becomes:

```python
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

    return res
```

### Mathematical Description of Cache Blocking

We start with  
$A\in\mathbb{R}^{N\times K}$,  
$B\in\mathbb{R}^{K\times M}$,  
$C\in\mathbb{R}^{N\times M}$.

Choose block sizes $(n_C,k_C,m_C)$ and panel sizes $(n_R,m_R)$.

1. **Partition into cache blocks**  
   For any element indices $(i,p,j)$ define block‐indices and remainders:
$$
     i_b = \left\lfloor \frac{i}{n_C}\right\rfloor,\quad
     r   = i \bmod n_C,
     \qquad
     p_b = \left\lfloor \frac{p}{k_C}\right\rfloor,\quad
     s   = p \bmod k_C,
     \qquad
     j_b = \left\lfloor \frac{j}{m_C}\right\rfloor,\quad
     t   = j \bmod m_C.
   $$
   (r, s, t mean in which cache block we are in)

   Then the cache blocks are
   $$
     \bar A_{i_b,p_b}[r,s]
     = A\bigl[i_b\,n_C + r,\;p_b\,k_C + s\bigr],
   $$
   $$
     \bar B_{p_b,j_b}[s,t]
     = B\bigl[p_b\,k_C + s,\;j_b\,m_C + t\bigr].
   $$

2. **Subdivide each block into panels**  
   Inside each $\bar A_{i_b,p_b}$ we form panels of height $n_R$:
   $$
     \bar A_{i_b,p_b}^{(r_b)}[u,s]
     = \bar A_{i_b,p_b}[\,r_b\,n_R + u,\;s\,],
     \quad
     u=0,\dots,n_R-1,
     \quad
     r_b=0,\dots,\Bigl\lceil\frac{n_C}{n_R}\Bigr\rceil-1.
   $$
   Inside each $\bar B_{p_b,j_b}$ we form panels of width $m_R$:
   $$
     \bar B_{p_b,j_b}^{(t_b)}[s,v]
     = \bar B_{p_b,j_b}[\,s,\;t_b\,m_R + v\,],
     \quad
     v=0,\dots,m_R-1,
     \quad
     t_b=0,\dots,\Bigl\lceil\frac{m_C}{m_R}\Bigr\rceil-1.
   $$

3. **Pack panels contiguously**  
   Let $\mathrm{vec}(\cdot)$ denote row‐major flattening. Then
   $$
     \widetilde A_{i_b,p_b}^{(r_b)}
     = \mathrm{vec}\bigl(\bar A_{i_b,p_b}^{(r_b)}\bigr)
     \;\in\;\mathbb{R}^{\,n_R\times k_C},
   $$
   $$
     \widetilde B_{p_b,j_b}^{(t_b)}
     = \mathrm{vec}\bigl(\bar B_{p_b,j_b}^{(t_b)}\bigr)
     \;\in\;\mathbb{R}^{\,k_C\times m_R}.
   $$

With these definitions, the BLIS‐style 5‐loop GEMM computes
$$
  C_{i,j}
  \;+\!=\;
  \sum_{p_b}
  \sum_{r_b,t_b}
    \bigl(\widetilde A_{i_b,p_b}^{(r_b)}\bigr)^{T}
    \,\widetilde B_{p_b,j_b}^{(t_b)},
$$
where 
$$
  i = i_b\,n_C + r_b\,n_R + u,\quad
  j = j_b\,m_C + t_b\,m_R + v.
$$
This reorganises the original triple‐sum
$$
  C_{i,j} \;+\!=\; \sum_{p=0}^{K-1} A_{i,p}\,B_{p,j}
$$
into cache‐block, panel and packed‐vector accesses.