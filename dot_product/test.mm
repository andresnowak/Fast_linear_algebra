#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <Metal/MTLCommandBuffer.h> // exposes GPUStartTime / GPUEndTime
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include "test.h"

std::pair<float, double> test_dot_product(const std::vector<float> &a, const std::vector<float> &b, float (*reduce_function)(size_t n, float *out), int gridSizeValues[3], int threadGroupSizeValues[3]) {
    // 1) Create the Metal device & queue

    // id means * (a pointer to an objc object, it is the universal object pointer type)
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();

    if (!device){
        std::cerr << "Error: this device doesn't support Metal\n";
        std::abort();
    }

    // commandQueue a serial queue of command buffers (like copy, compute, render, etc..)
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // 2) Host data
    size_t n = a.size(), byteCount = n * sizeof(float);

    // 3) Create the GPU buffers
    // id<MTLBuffer> bufA = [device newBufferWithBytes:A.data() length:byteCount options:MTLResourceStorageModeShared];
    id<MTLBuffer> sharedA = [device newBufferWithBytes:a.data() length:byteCount options:MTLResourceStorageModeShared];
    id<MTLBuffer> sharedB = [device newBufferWithBytes:b.data() length:byteCount options:MTLResourceStorageModeShared];
    id<MTLBuffer> sharedOut = [device newBufferWithLength:byteCount options:MTLResourceStorageModeShared];

    id<MTLBuffer> bufA = [device newBufferWithLength:byteCount options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bufB = [device newBufferWithLength:byteCount options:MTLResourceStorageModePrivate];
    id<MTLBuffer> bufOut = [device newBufferWithLength:byteCount options:MTLResourceStorageModePrivate];


    // Start command queue
    MTLCommandBufferDescriptor *desc = [[MTLCommandBufferDescriptor alloc] init];
    desc.retainedReferences = YES;

    id<MTLCommandBuffer> cmd = [queue commandBufferWithDescriptor:desc];

    // Copy the data to the private buffers
    id<MTLBlitCommandEncoder> copyEnc = [cmd blitCommandEncoder];

    [copyEnc copyFromBuffer:sharedA sourceOffset:0 toBuffer:bufA destinationOffset:0 size:byteCount];
    [copyEnc copyFromBuffer:sharedB sourceOffset:0 toBuffer:bufB destinationOffset:0 size:byteCount];
    
    [copyEnc endEncoding];


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
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufOut offset:0 atIndex:2];

    MTLSize gridSize = MTLSizeMake(gridSizeValues[0], gridSizeValues[1], gridSizeValues[2]); // (x, y, z)
    MTLSize threadgroupSz = MTLSizeMake(threadGroupSizeValues[0], threadGroupSizeValues[1], threadGroupSizeValues[2]); // (x, y, z)
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSz];
    [enc endEncoding];

    // Copy the result data
    copyEnc = [cmd blitCommandEncoder];
    [copyEnc copyFromBuffer:bufOut sourceOffset:0 toBuffer:sharedOut destinationOffset:0 size:byteCount];
    [copyEnc endEncoding];

    // Run
    auto start = std::chrono::high_resolution_clock::now();
    [cmd commit];
    [cmd waitUntilCompleted];

    // 6) Read back & reduce
    float *results = (float*)sharedOut.contents;

    float dot = 0;
    if (reduce_function != NULL) {
        dot = reduce_function(n, results);
    } else {
        // TODO: grab values directly from result
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto wallClockTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    double gpuStart = cmd.GPUStartTime;   // seconds
    double gpuEnd   = cmd.GPUEndTime;     // seconds
    double kernelTime    = (gpuEnd - gpuStart) * 1e9;   // nanoseconds

    std::cout << gpuNs << std::endl;

    return {dot, kernelTime};
}