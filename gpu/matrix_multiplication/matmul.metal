#include <metal_stdlib>
using namespace metal;

typedef struct{
    uint N;
    uint K;
    uint M;
} Sizes;

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