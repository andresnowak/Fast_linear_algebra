#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <iostream>
#include <vector>

int main() {
    // 1) Create the Metal device & queue
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        std::cerr << "Error: this device does not support Metal\n";
        return -1;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // 2) Host data
    std::vector<float> A{1,2,3,4}, B{5,6,7,8};
    size_t n = A.size(), byteCount = n * sizeof(float);

    // 3) Create GPU buffers
    id<MTLBuffer> bufA   = [device newBufferWithBytes:A.data()
                                               length:byteCount
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB   = [device newBufferWithBytes:B.data()
                                               length:byteCount
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufOut = [device newBufferWithLength:byteCount
                                              options:MTLResourceStorageModeShared];

    // 4) Load the pre–compiled .metallib
    NSError *error = nil;
    NSURL *libURL = [NSURL fileURLWithPath:@"dot_product.metallib"];
    id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&error];

    if (!lib) {
        std::cerr << "Library load error: "
                  << [[error localizedDescription] UTF8String] << "\n";
        return -1;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"dotProduct"];
    id<MTLComputePipelineState> pipeline =
      [device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        std::cerr << "Pipeline creation error: "
                  << [[error localizedDescription] UTF8String] << "\n";
        return -1;
    }

    // 5) Encode & dispatch
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA   offset:0 atIndex:0];
    [enc setBuffer:bufB   offset:0 atIndex:1];
    [enc setBuffer:bufOut offset:0 atIndex:2];

    MTLSize gridSize      = MTLSizeMake(n, 1, 1);
    MTLSize threadgroupSz = MTLSizeMake(1, 1, 1);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSz];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    // 6) Read back & reduce
    float *results = (float*)bufOut.contents;
    float dot = 0;
    for (size_t i = 0; i < n; ++i) dot += results[i];

    std::cout << "Inner Product: " << dot << "\n";
    return 0;
}