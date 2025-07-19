#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        // 1) Create the Metal device & queue
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"Error: this device does not support Metal");
            return -1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // 2) Host data
        float A[] = {1, 2, 3, 4};
        float B[] = {5, 6, 7, 8};
        size_t n = sizeof(A) / sizeof(A[0]);
        size_t byteCount = n * sizeof(float);

        // 3) Create GPU buffers
        id<MTLBuffer> bufA   = [device newBufferWithBytes:A
                                                   length:byteCount
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufB   = [device newBufferWithBytes:B
                                                   length:byteCount
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufOut = [device newBufferWithLength:byteCount
                                                  options:MTLResourceStorageModeShared];

        // 4) Load the pre-compiled .metallib
        NSError *error = nil;
        NSURL *libURL = [NSURL fileURLWithPath:@"dot_product.metallib"];
        id<MTLLibrary> lib = [device newLibraryWithURL:libURL error:&error];
        if (!lib) {
            NSLog(@"Library load error: %@", error.localizedDescription);
            return -1;
        }

        id<MTLFunction> fn = [lib newFunctionWithName:@"dotProduct"];
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:fn error:&error];
        if (!pipeline) {
            NSLog(@"Pipeline creation error: %@", error.localizedDescription);
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
        float *results = (float *)bufOut.contents;
        float dot = 0.0f;
        for (size_t i = 0; i < n; ++i) dot += results[i];

        NSLog(@"Inner Product: %f", dot);
    }
    return 0;
}