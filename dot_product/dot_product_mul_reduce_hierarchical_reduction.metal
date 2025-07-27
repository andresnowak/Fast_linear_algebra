#include <metal_stdlib>
using namespace metal;

kernel void dotProduct(const device float* a [[ buffer (0) ]],
                        const device float* b [[ buffer (1) ]],
                        device atomic_float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]],
                        uint3 lid [[ thread_position_in_threadgroup ]]
                        ) {

    threadgroup float shared[1024];
    shared[lid.x] = a[tid.x] * b[tid.x];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpt.x / 2; stride >= 1; stride /= 2) {
        if (lid.x < stride) {
            shared[lid.x] += shared[lid.x + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    
    if (lid.x == 0) {
        atomic_fetch_add_explicit(&out[0], shared[lid.x], memory_order_relaxed);
    }
}
