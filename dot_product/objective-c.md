## Language things

Objective-c behaves like small talk where we do message passing and we are dynamic instead of having functions be static `fnName(arguments)`
- It looks in a table of the class for the function to pass the message, we call functions with `[]`, and this lookup happens at runtime instead at compile time like c where the address is hardcoded here we have to look for the address at runtime in the table.
- So here we have methods no, and they look like this in a way `newBufferWithBytes:length:options:` so here we are passing three arguments `[device newBufferWithBytes:A.data() length:byteCount options:MTLResourceStorageModeShared]` to device (and object created from class MTLDevice) and we expect the object to respond to this message
- `id` keyword means pointer (`*` in c)


## Metal things

- `commandQueue` a serial queue of command buffers (like copy, compute, render, etc..)
- Then we have `[queue commandBuffer]`, this commandBuffer is a container for the GPU work that will be submitted together, then we have the encoder `id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];` this is what lets us *record the commands* into the buffer (like a GPU *instruction stream*.)
- Then we have the pipeline `[enc setComputePipelineState:pipeline];` this one contains the micro-code of the kernel (here in this case doProduct) and it also contains the static values (the state for this kernel like register count, thread width, etc..)
- Then we have `[enc setBuffer:bufA offset:0 atIndex:0];, [enc setBuffer:bufB offset:0 atIndex:1];, [enc setBuffer:bufOut offset:0 atIndex:2];` here we are assigning our three buffers (bufA, bufB and bufOut) to the buffers we defined in our kernel, here the index means the values we put on the buffer for each argument in our dotProduct kernel (we are assigining each buffer to its corresponding buffer argument in the kernel)
- Then we have `MTLSize gridSize = MTLSizeMake(n, 1, 1);` and `MTLSize threadgroupSz = MTLSizeMake(1, 1, 1);`. The first one says how many workers we want in a grid (the global size or dispatch size) and the threadgroupSz tells us how we want those workers group together (this is also called local size or work group size)
  - The `gridSize` can be defined as a 3D index space (for convenience to the programmer)
    - The `gridSize` only says how many threads we want
    - This says basically how many threads we want to use for a piece of data
    - So basically if i have vector of 10_000 values to work with, one has to declare a grid size of (10_000, 1, 1) (and then we divide this into groups)
  - The the `threadGroup` (or compute unit) is a subdivision of the logical grid we choose 
    - it has to be less than the amount of threads we have (here we have 512)
    - This basically does the division of our gridSize for how to use the memory, but the driver will still define the warps to use, but like if our grid size is less than 32 threads we are basically wasting a warp on this
    - The `threadGroup` also helps us define the barrier synchronization and lets schedule efficiently the work on fixed simd units
    - This also creates what is called a `threadGroup memory` , this is the cache we control that only this threads can see
- Then `[enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSz];`  schedules the kernel to run once for each element (`gridSize`) and how many times (`threadgroupSz`)
- Then here `[enc endEncoding];` we finish encoding the compute commands (so just creating the pipeline (the kernel) and dispatching the threads)
- Then `[cmd commit];` we commit the work to run on the GPU
- And finally `[cmd waitUntilCompleted];` we tell the cpu to wait until the GPU has finished 