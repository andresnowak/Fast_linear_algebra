#include <metal_stdlib>
using namespace metal;

typedef struct{
    uint N;
    uint K;
    uint M;
} Sizes;

#define TILE_SIZE 32

kernel void matmul(const device float* A [[ buffer(0) ]],
                    const device float* B [[ buffer(1) ]],
                    device float* C [[ buffer(2) ]],
                    const device Sizes& params [[ buffer(3) ]],
                    uint3 gid [[ thread_position_in_grid ]],
                    uint3 tpt [[ threads_per_threadgroup ]],
                    uint3 lid [[ thread_position_in_threadgroup ]]) {
    threadgroup float Mds[TILE_SIZE][TILE_SIZE]; // Here tile size is the size of the threadgroup
    threadgroup float Nds[TILE_SIZE][TILE_SIZE];

    uint row = gid.y;
    uint col = gid.x;

    float sum = 0.0;

    for (int pk = 0; pk < (params.K + TILE_SIZE - 1) / TILE_SIZE; ++pk) {
        // Here lid.x and lid.y are the offsets when moving our tiles in the K dimension
        if (row < params.N && (TILE_SIZE * pk + lid.x) < params.K) {
            Nds[lid.y][lid.x] = A[row * params.K + (TILE_SIZE * pk + lid.x)];
        }
        else {
            Nds[lid.y][lid.x] = 0.0;
        }
        if ((pk * TILE_SIZE + lid.y) < params.K && col < params.M) {
            Mds[lid.y][lid.x] = B[(pk * TILE_SIZE + lid.y) * params.M + col];
        }
        else {
            Mds[lid.y][lid.x] = 0.0;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup); // True dependence

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += Nds[lid.y][k] * Mds[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup); // False dependence
    }

    if (row < params.N && col < params.M) {
        C[row * params.M + col] = sum;     
    }
} 