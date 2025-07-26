#include <metal_stdlib>
using namespace metal;

kernel void dotProduct(const device float* a [[ buffer (0) ]],
                        const device float* b [[ buffer (1) ]],
                        device atomic_float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]],
                        uint3 lid [[ thread_position_in_threadgroup ]]
                        ) {

    threadgroup float shared[1024]; // threadgroup is the storage qualifier saying this memory lives in sram (cache, flip-flop), that is shared by all threads (I think we should use variable tip here, but we need to use constants, and it seems tpt is not constant)

    float prod = a[tid.x] * b[tid.x];

    shared[lid.x] = prod;

    threadgroup_barrier(mem_flags::mem_threadgroup); // Wait for all threads in threadgroup to finish

    if (lid.x == 0) { // First thread does the reduction
        float sum = 0.0;

        for (uint i = 0; i < tpt.x; ++i) {
            sum += shared[i];
        }

        atomic_fetch_add_explicit(&out[0], sum, memory_order_relaxed);
    }
}