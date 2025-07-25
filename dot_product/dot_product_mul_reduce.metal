#include <metal_stdlib>
using namespace metal;

kernel void dot_product(const device float* a [[buffer (0)]],
                        const device float* b [[buffer (1)]],
                        device float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]]
                        uint3 tpt [[ threads_per_threadgroup ]]
                        uint3 lid [[ thread_position_in_threadgroup ]]
                        ) {
    threadgroup float shared[tpt.x]; // threadgroup is the storage qualifier saying this memory lives in sram (cache, flip-flop), that is shared by all threads (I think we should use variable tip here)

    


}
