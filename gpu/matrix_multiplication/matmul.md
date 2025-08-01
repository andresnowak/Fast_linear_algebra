# Matmul

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

This version is very naive as we have to know that values from A and B matrix are used multiple times, if our threadgroups were of size 2x2, we would be loading each value from Matrix A and B two times, if it was 4x4 then 4 times each, etc.

Here the Arithmetic intensity is $\frac{2 \text{Flops}}{8 \text{bytes} = 0.25 \text{Op/B}} (8 bytes because we load 2 times a value of size float32 so 4 bytes each) This is a very bad ratio, because we are only doing 0.25 Ops per byte, here we are Memory bounded
- If our Gpu had a compute bound of 19.5 TFLOP/s and a memory bandwith of 1.6 TB/s, Our performance here would be 
  - Perf = $\min(\text{Peak compute}, \text{A.I.} \cdot \text{Peak bandwith}) = \min(19.5, 1600 * 0.25) = 400 \text{GFLOP/s}$