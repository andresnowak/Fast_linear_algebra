# Dot product

CPU and GPU memory in M processors is unified

## Dot product mul GPU and reduce on CPU

Here first we are passing two vectors $a$ and $b$ that have its buffers binded to buffers indices 0, 1 and 2 (At least this i s a way where we can pass the data to our kernels) 

Then in our metal kernel we will be doing the multiplication of of our two vectors and the result of each scalar multiplication will be put in its corresponding index in a out buffer

```c++
kernel void dotProduct(const device float* a [[ buffer(0) ]],
                         const device float* b [[ buffer(1) ]],
                         device float* out [[ buffer(2) ]],
                        uint3 id [[ thread_position_in_grid ]]) {
    out[id.x] = a[id.x] * b[id.x];                        
}
```

First what is the thread_position_in_grid? here we can see how instead of doing a loop as we commonly know
```c++
for (int i = 0; i < n; i++) {
    out[i] = a[id] * b[id]
}
```

We are instead doing parallel group between what we call grids

```objective-c
MTLSize gridSize = MTLSizeMake(n, 1, 1); // (x, y, z)
MTLSize threadgroupSz = MTLSizeMake(256, 1, 1); // (x, y, z)
[enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSz]
```

- What we are doing here is that first we are going to create a 3D grid as a 1 dimensional line of n values (lets say 4096 values) ```MTLSize gridSize = MTLSizeMake(n, 1, 1);``` (because this is our total amount of values in our vectors)
- Then we will divide this grid into our threadGroups (this can't be bigger than the amount of threads we have, in this case we have 2048 (or 1024 not completely sure) threads in the M1 pro)
  - This thread groups will have a shared memory
    - Based on this the best thread groups sizes here are around 256 because 
- Dot-product is a very compute bound problem (as there is very little operations we can do)
  - Using (256 threads = 8 warps (of 32 threads each) = 2 warps per SIMD * 4 SIMDs)
  - using 4 warps wouldn't help





For this simple one for now we will be doing the reduction on the CPU

```c++
  auto reduce = [](size_t n, float* results) -> float {
      float dot = 0;

      #pragma omp parallel for reduction(+:dot)
      for (size_t i = 0; i < n; ++i) {
          dot += results[i];
      }

      return dot;
  };
```