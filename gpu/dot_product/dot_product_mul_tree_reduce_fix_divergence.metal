#include <metal_stdlib>
using namespace metal;

kernel void dotProduct(const device float* a [[ buffer (0) ]],
                        const device float* b [[ buffer (1) ]],
                        device float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]]
                        ) {

    threadgroup float shared[1024];
    shared[tid.x] = a[tid.x] * b[tid.x];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint position = tid.x;

    for (uint stride = tpt.x; stride >= 1; stride /= 2) {
        if (tid.x < stride) {
            shared[position] += shared[position + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid.x == 0) {
        out[0] = shared[tid.x];
    }
}
