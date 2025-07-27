# Dot product

- CPU and GPU memory in M processors is unified
- Also in this examples we are not doing dot-products that optimize for different sizes and we are also not doing a check for if the size of the vector is less than the amount of threads we have (because we would be accessing out of bounds but in metal i think the scheduler already does this check with how many threads it can launch based on grid size, the scheduler already mask the unused SIMD lanes)
- Finally **All the algorithms here will run with an input of size 1024**, we do this because of how the reduction algorithms work
  - We want to use a threadgroup of size 1024 to be able to then do the tree reduce type algorithms on this with the gpu (so input = 2 * threadgroup), but because we also need to do first vector multiplication (*so here we would need two threadgroups instead of 1, or the same thread multiplying its value and the next one*) and we can't synchronize across threadgroups we instead use inputs of 1024. We do this so we can do the whole reduction with one block to the scalar, if we don't have this it is necessary to do reduction on cpu, do atomic add across blocks on the GPU, or launch the kernel multiple times to reduce the size of vector until we get just a scalar.
- *For the threadgroup sizes, I'm still confused as we can use 1024 threads okay, but all this threads can't run in a block in parallel no? as one core can only do 128 threads (and maybe can do multiple warps in one cycle)*

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

We are instead iterating our vector (or better said our defined grid) with threads (so defining parallel work from the get go)

```objective-c
MTLSize gridSize = MTLSizeMake(n, 1, 1); // (x, y, z)
MTLSize threadgroupSz = MTLSizeMake(1024, 1, 1); // (x, y, z)
[enc dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSz]
```

- What we are doing here is that first we are going to create a 3D grid as a 1 dimensional line of n values (lets say 4096 values) ```MTLSize gridSize = MTLSizeMake(n, 1, 1);``` (because this is our total amount of values in our vectors)
- Then we will divide this grid into our threadGroups (this can't be bigger than the amount of threads we have 1024 threads available to use use in one core (because 4SIMDs x 32 threads each x waves (warps) per SIMD) threads in the M1 pro)
  - This thread groups will have a shared memory
- Dot-product is a very compute bound problem (as there is very little operations we can do)


And for this simple kernel we only do the multiplication and the reduction we will do it on the CPU side, by copying our out result vector to CPU and then doing the reduction in parallel with omp

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

**Cost: 3N Global loads (because we do load in gpu for a and b, and then in cpu load for outs), N muls, and N sums**

Here we are basically dividing our for loop into chucks for each thread in the CPU, saying that each cpu gets its own private copy of the dot variable, and then we add all results together atomically in the global dot variable

## Dot product mul reduce GPU and final reduce CPU

Here will be our first version where we start to do some work of the reduce on the GPU

```c++
kernel void dotProduct(const device float* a [[ buffer (0) ]],
                        const device float* b [[ buffer (1) ]],
                        device float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]],
                        uint3 lid [[ thread_position_in_threadgroup ]]
                        ) {

    threadgroup float shared[1024]; // threadgroup is the storage qualifier saying this memory lives in sram (cache, flip-flop), that is shared by all threads (I think we should use variable tip here, but we need to use constants, and it seems tpt is not constant)

    float prod = a[tid.x] * b[tid.x];

    shared[lid.x] = prod;

    threadgroup_barrier(mem_flags::mem_threadgroup); // Wait for all threads in threadgroup to finish

    if (lid.x == 0) { // First thread does the reduction
        float sum = 0.0;

        for (uint i = 0; i < tpt.x; ++i) {
            sum += shared[i];
        }

        out[tid.x / tpt.x] = sum;
    }
    
}
```

- Here now we are creating first a vector memory in SRAM (the cache) for our threadgroup, here all the threads in a threadgroup will put their multiplication value.
- Then we make the threadgroup wait for all threads to finish, 
- And then we do an accumulation with the thread with id 0 of that threadgroup and we write to basically the threadgroup number, basically we want the values to be contiguous in this out vector so the cpu can more easily do the last reduction, because here the only thing we are doing is reducing a little bit of the work for the cpu

```c++
  auto reduce = [](size_t n, float* results) -> float {
    float dot = 0;
  
    #pragma omp parallel for reduction(+:dot)
    for (size_t i = 0; i < ceil(n / 1024.0); ++i) {
        dot += results[i];
    }

    return dot;
  };
```

But we can see this example is slower than our first version and it makes sense because first we are doing a threadgroup barrier waiting to then do just part of the work for then the cpu to finish.

Compared to the other version that just does the multiplication on the GPU and then does parallel reduction on the CPU, here we have basically now more loads (even though they are from SRAM) compared to the older version

**Cost: $N Global loads (because we do load in gpu for a and b and then again from our shared memory, and then in cpu load for outs), N muls, and N sums**


## Dot product mul reduce GPU

For this one we will now do the whole reduction on the GPU kernel by doing an atomic add


```c++
kernel void dotProduct(const device float* a [[ buffer (0) ]],
                        const device float* b [[ buffer (1) ]],
                        device atomic_float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]],
                        uint3 lid [[ thread_position_in_threadgroup ]]
                        ) {

    threadgroup float shared[1024]; // threadgroup is the storage qualifier saying this memory lives in sram (cache, flip-flop), that is shared by all threads (I think we should use variable tip here, but we need to use constants, and it seems tpt is not constant)

    float prod = a[tid.x] * b[tid.x];

    shared[lid.x] = prod;

    threadgroup_barrier(mem_flags::mem_threadgroup); // Wait for all threads in threadgroup to finish

    if (lid.x == 0) { // First thread does the reduction
        float sum = 0.0;

        for (uint i = 0; i < tpt.x; ++i) {
            sum += shared[i];
        }

        atomic_fetch_add_explicit(&out[0], sum, memory_order_relaxed);
    }
}
```

*Also see here that we are using atomic_float, but we are casting our output float vector (pointer) to an atomic float, this is possible because our memory is already correctly aligned for the float and we only modify for this case the values with atomic operations, the only other operation we do is a copy to cpu and thats it*

## Dot product mul tree reduce GPU

Now lets implement tree reduce algorithm, here for the first version as how the algorithm is defined, we have that each thread will reduce two values, the value at its position and the one next to it.

So here the thread positions will be defined by `uint position = thread_position_in_grid * 2`

```c++
kernel void dotProduct2(const device float* a [[ buffer (0) ]],
                        const device float* b [[ buffer (1) ]],
                        device float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]]
                        ) {
    out[tid.x] = a[tid.x] * b[tid.x];

    threadgroup_barrier(mem_flags::mem_device); // Wait for all threads in threadgroup to finish and memory is ordered, so basically saying which values is the one we will be able to use (so we linearize the operations thats what it means), because if not it is possible to have the regular register problem where a first read can read the new value and next read by another thread can read an old value (we need for each thread to see the same order in memory)

    uint position = tid.x * 2;

    for (uint stride = 1; stride <= tpt.x; stride *= 2) {
        if (tid.x % stride == 0) {
            out[position] += out[position + stride];
        }

        threadgroup_barrier(mem_flags::mem_device);
    }
}
```

Here we do the work but directly on the memory device (so we work in DRAM and then L2 and L1 cache *and i think the coherency would be directly in cache, and we don't need to go to DRAM, because the the threadgroup will probably share the same cache*).

But instead of doing it this way we can use the threadgroup shared memory (that is of type SRAM) that is even faster than the L1 cache (and smaller, I think 32 kb here), this way we can work faster and can have better synchronization

```c++
kernel void dotProduct(const device float* a [[ buffer (0) ]],
                        const device float* b [[ buffer (1) ]],
                        device float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]]
                        ) {

    threadgroup float shared[1024];
    shared[tid.x] = a[tid.x] * b[tid.x];

    threadgroup_barrier(mem_flags::mem_threadgroup);

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
```

## Dot product mul tree reduce fix control divergence GPU

Now for this method instead of a thread reducing two contiguous values (its position and the one next to it), instead we will have each thread add the value at its position and one that is a block away (a threadgroup away) from it. 

- This way we do two things
  - We have made better coalesced memory access, because now threads are accessing values next to each other so they are using the values of the cache lane.
  - And we have removed control divergence, because here a warp is either executing a block or not (only if the input is not exactly divisible by the block size then we would have one warp at least with divergence at each point)

```c++
kernel void dotProduct(const device float* a [[ buffer (0) ]],
                        const device float* b [[ buffer (1) ]],
                        device float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]]
                        ) {

    threadgroup float shared[1024];
    shared[tid.x] = a[tid.x] * b[tid.x];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint position = tid.x;

    for (uint stride = tpt.x; stride >= 1; stride /= 2) {
        if (tid.x < stride) {
            shared[position] += shared[position + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid.x == 0) {
        out[0] = shared[tid.x];
    }
}
```

## Dot product mul tree reduce hierarchical reduction GPU

Now if you saw the other versions of tree reduce the problem with them is that they assume that the input will be double the size of the largest threadgroup we can use (so in this case 2048, and because we are doing the dot product we use 1024 because we need each thread to do the multiplication, but even though we could have done it so the thread does two outputs of the multiplications)

To fix this we do now something called hierarchical reduction, where first each threadgroup does a reduction on its own shared memory and then the results of this reduced vector will be added to the first position of the output vector with an atomic add so as to have linearity in the operations between threadgroups for the final reduction into a scalar (it is not possible to do synchronization between threadgroups)

```c++
kernel void dotProduct(const device float* a [[ buffer (0) ]],
                        const device float* b [[ buffer (1) ]],
                        device atomic_float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]],
                        uint3 lid [[ thread_position_in_threadgroup ]]
                        ) {

    threadgroup float shared[1024];
    shared[lid.x] = a[tid.x] * b[tid.x];

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpt.x / 2; stride >= 1; stride /= 2) {
        if (lid.x < stride) {
            shared[lid.x] += shared[lid.x + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    
    if (lid.x == 0) {
        atomic_fetch_add_explicit(&out[0], shared[lid.x], memory_order_relaxed);
    }
}
```

## Extra

### Thread coarsening

Now there is a last trick for reductions, the last idea is why do we have to reduce only two values per thread, why not three or four or whatever

That is the idea here where we first do a reduction for a single thread for all the values it will work on and then we reduce per block and finally we reduce across blocks

- This method can be faster because now each thread can do a little bit more floating operations, so we are doing more computations and we are also reducing the size of our final vector where we will have in the end reduce across the blocks their results in global memory (so less atomicAdds)

The thing is here in the dot_product we don't implement it, because the thing is that here we first depend on the result of the multiplication of the two vectors and there is no way to do synchronization across threadgroups, and this technique helps when doing the first reduction form the input vector (in this case the output one where are writing the vector mul result), and form this we reduce the final size of our vector (because each block did more work in the end) and we have now less atomic adds across blocks.

But an implementation in metal would look like this for the thread coarsening reduce

```c++
#define COARSE_FACTOR 2

kernel void reduce(const device float* a [[ buffer (0) ]],
                        device atomic_float* out [[buffer (2)]],
                        uint3 tid [[ thread_position_in_grid ]],
                        uint3 tpt [[ threads_per_threadgroup ]],
                        uint3 lid [[ thread_position_in_threadgroup ]],
                        uint3 size [[ threads_per_grid ]])
                        ) {
    
    threadgroup float shared[1024];

    float sum = 0.0;

    #pragma unroll
    for (uint tile = 1; tile <= COARSE_FACTOR; ++tile) {
      int index = tid.x + tpt.x * tile;
      if (index < size.x) {
        sum += a[index];
      }
    }

    shared[lid.x] = sum;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpt.x / 2; stride >= 1; stride >>= 1) {
        if (lid.x < stride) {
            shared[lid.x] += shared[lid.x + stride];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    
    if (lid.x == 0) {
        atomic_fetch_add_explicit(&out[0], shared[lid.x], memory_order_relaxed);
    }
}
```

with a coarse factor of 2 we are doing basically the same as we where doing before in reality (or well in this case not because we only used 1024 values (size of threadgroup) instead of 2048 at first by grabbing directly from the input because we need to first do the dot product)