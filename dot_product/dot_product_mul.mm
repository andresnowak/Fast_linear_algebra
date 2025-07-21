#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>

#include "dot_product.h"

std::pair<float, double> dot_product_mul(const std::vector<float> &A, const std::vector<float> &B) {
    // 1) Create the Metal device & queue

    // id means * (a pointer)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (!device){
        std::cerr << "Error: this device doesn't support Metal\n";
        std::abort();
    }

    // commandQueue a serial queue of command buffers (like copy, compute, render, etc..)
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // 2) Host data
    size_t n = A.size(), byteCount = n * sizeof(float);

    // 3) Create the GPU buffers
    id<MTLBuffer> bufA = [device newBufferWithBytes:A.data() length:byteCount options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [device newBufferWithBytes:B.data() length:byteCount options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufOut = [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];

    // 4) Load the pre-compiled .metallib (the metal kernel)
    NSError *error = nil;
    NSURL *libURL = [NSURL fileURLWithPath:@"dot_product.metallib"];
    id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&error];

    if (!lib) {
        std::cerr << "Library load error: " << [[error localizedDescription] UTF8String];

        std::abort();
    }

    id<MTLFunction> dotProductFn = [lib newFunctionWithName:@"dotProduct"];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:dotProductFn error:&error];

    if (!pipeline) {
        std::cerr << "Pipeline creation error: " << [[error localizedDescription] UTF8String];

        std::abort();
    }

    // 5) Encode & dispatch
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufOut offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(n, 1, 1); // (x, y, z)
    MTLSize threadgroupSz = MTLSizeMake(n, 1, 1); // (x, y, z)
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSz];
    [enc endEncoding];

    // Run
    auto start = std::chrono::high_resolution_clock::now();
    [cmd commit];
    [cmd waitUntilCompleted];

    // 6) Read back & reduce
    float *results = (float*)bufOut.contents;
    float dot = 0;
    for (size_t i = 0; i < n; ++i) {
        dot += results[i];
    }
    auto end = std::chrono::high_resolution_clock::now();

    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    return {dot, ns.count()};
}
