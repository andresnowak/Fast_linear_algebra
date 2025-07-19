#include <metal_stdlib>
using namespace metal;


// A single‐pass inner‐product + reduction in one threadgroup.
//   - A, B: input vectors of length N
//   - out:  single‐float output buffer (length 1)
//   - N:    vector length
//   - sdata: dynamic threadgroup memory of N floats
kernel void dotProduct(const device float* a [[ buffer(0) ]],
                         const device float* b [[ buffer(1) ]],
                         device float* out [[ buffer(2) ]],
                        uint id [[ thread_position_in_grid ]]) {
    out[id] = a[id] * b[id];                        
}