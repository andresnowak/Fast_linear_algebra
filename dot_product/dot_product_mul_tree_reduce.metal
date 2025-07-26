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

    threadgroup_barrier(mem_flags::mem_threadgroup); // Wait for all threads in threadgroup to finish and memory is ordered, so basically saying which values is the one we will be able to use (so we linearize the operations thats what it means), because if not it is possible to have the regular register problem where a first read can read the new value and next read by another thread can read an old value (we need for each thread to see the same order in memory)

    uint position = tid.x * 2;

    for (uint stride = 1; stride <= tpt.x; stride *= 2) {
        if (tid.x % stride == 0) {
            shared[position] += shared[position + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid.x == 0) {
        out[0] = shared[tid.x];
    }
}
