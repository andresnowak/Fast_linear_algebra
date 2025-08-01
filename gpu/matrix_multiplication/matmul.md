# Matmul

### Peak FP32 throughput of the M1 Pro

Given  
- 256 FP32 FLOPs per **core** per clock  
- 1.296 GHz base clock  
- 16 cores  

```text
Peak GFLOP/s = 256 FLOPs/clk/core × 1.296 GHz × 16 cores
              = 256 × 1.296 × 16
              = 5 304.576 GFLOP/s
              ≈ 5.30 TFLOP/s
```

Therefore the **maximum theoretical FP32 performance** is  
**≈ 5.30 TFLOP/s**.

## Matmul naive
For the first version of the matmul we make so that each thread will calculate an output from Matrix C, basically one thread will iterate all the row from Matrix A and column Matrix B and do the dot product of both to compute its value in the C matrix at its position in the grid

```c++
kernel void matmul(const device float* A [[ buffer(0) ]],
                    const device float* B [[ buffer(1) ]],
                    device float* C [[ buffer(2) ]],
                    const device Sizes& params [[ buffer(3) ]],
                    uint3 gid [[ thread_position_in_grid ]],
                    uint3 tpt [[ threads_per_threadgroup ]]) {

    float sum = 0.0;
    for (uint k = 0; k < params.K; ++k) {
        sum += A[gid.y * params.K + k] * B[k * params.M + gid.x];
    }  

    C[gid.y * params.M + gid.x] = sum;              
} 
```

This version is very naive as values from A and B matrix are used multiple times, for each operation that is one floating-point multiplication and one one floating point addition we have 2 global loads (one for A and one B value in the matrix)

Here the Arithmetic intensity is $\frac{2 \text{Flops}}{8 \text{bytes}} = 0.25 \text{Op/B}$ (8 bytes because we load 2 times a value of size float32 so 4 bytes each and we do only 2 operations a multiplication and a sum for each value of C) This is a very bad ratio, because we are only doing 0.25 Ops per byte, here we are Memory bounded
- If our GPU had a compute bound of 19.5 TFLOP/s and a memory bandwith of 1.6 TB/s, Our performance here would be 
  - Perf = $\min(\text{Peak compute}, \text{A.I.} \cdot \text{Peak bandwith}) = \min(19.5, 1600 * 0.25) = 400 \text{GFLOP/s}$
- **For our Matmul naive we get** we get a speed of $361.27$ GFLOPS (Our theoretical speed is 7.22 faster than this)


## Matmul Tiled

Now as we said before for the *Matmul Naive*, for a row in matrix A each of its values will be loaded M times (the size of the row in Matrix C (x direction)) and for a column in in matrix B each of its values will be loaded N times (the size of the column in Matrix C (y direction)) 

Now lets see what happens when we reutilize loads using tiles of work (our threadgroups)
- First we map a tile (a threadgroup) to a point in Matrix C, and we will move this tile in the K dimension in the A matrix and B matrix. 
- Now inside this tile if a row in the tile of Matrix C each thread will want to load the the matrix A row and matrix B column why don't we make it so that each thread loads one value from the A and B matrix and saves it in a shared memory, we will call them Nds for A and Mds for B.
- Then after all threads load a value we synchronize and then now each thread will do its own multiplication and accumulation in their on private sum accumulation variable
- Finally after this we synchronize again and we repeat moving our tile in the K dimension of both matrices and we do this until we finish and then we save our private sum registers on the corresponding position of the thread in the grid on the C matrix

```c++
kernel void matmul(const device float* A [[ buffer(0) ]],
                    const device float* B [[ buffer(1) ]],
                    device float* C [[ buffer(2) ]],
                    const device Sizes& params [[ buffer(3) ]],
                    uint3 gid [[ thread_position_in_grid ]],
                    uint3 tpt [[ threads_per_threadgroup ]],
                    uint3 lid [[ thread_position_in_threadgroup ]]) {
    threadgroup float Mds[TILE_SIZE][TILE_SIZE];
    threadgroup float Nds[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0;

    for (int pk = 0; pk < params.K / TILE_SIZE; ++pk) {
        // Here lid.x and lid.y are the offsets when moving our tiles in the K dimension
        Nds[lid.y][lid.x] = A[row * params.K + (TILE_SIZE * pk + lid.x)];
        Mds[lid.y][lid.x] = B[(pk * TILE_SIZE + lid.y) * params.M + col];

        threadgroup_barrier(mem_flags::mem_threadgroup); // True dependence

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Nds[lid.y][k] * Mds[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup); // False dependence
    }

    C[gid.y * params.M + gid.x] = sum;   
```