#include <metal_stdlib>
using namespace metal;


//   - A, B: input vectors of length N
//   - out:  out vector of length N
kernel void dotProduct(const device float* a [[ buffer(0) ]],
                         const device float* b [[ buffer(1) ]],
                         device float* out [[ buffer(2) ]],
                        uint3 id [[ thread_position_in_grid ]]) {
    out[id.x] = a[id.x] * b[id.x];                        
}

// kernel void dotProduct(const device float* a [[ buffer(0) ]],
//                          const device float* b [[ buffer(1) ]],
//                          device float* out [[ buffer(2) ]],
//                         uint id [[ thread_position_in_grid ]]) {
//     out[id] = a[id] * b[id];                        
// }