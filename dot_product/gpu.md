Programming massively parallel processors

## CPU
1. So first in a cpu the design is more targeted to single threaded with the idea of the control logic where this allows to execute things in out of order or even in parallel in a single thread.
2. And we also have the data and instruction caches to reduce the latencies of accessing the instruction and data from ram 

## GPU
1. A gpu is more designed to parallel computation
2. A gpu has a bigger bandwith compared to the cpu (this is because of necessity for things like the graphic frame buffer requirements)
3. gpus can have a higher arithmetic intensity as one needs 8x more computations per byte transfer $\text{A.I} = \frac{\text{FLOPs}}{\text{Total data movements (Bytes)}}$
   1. A low arithmetic intensity means the the program is spending more time transfering data than performing computations (This suggests it is memory bound, that it has limited memory bandwith, so it has to do more data movements)
   2. Majority of algorithms can have the theoretical Arithmetic intensity of the GPUs, but some exceptions are things like Matmul 
4. For M series processors, using the shared memory makes it so we don't need to copy the data, but it still adds overhead
   1. Like having the memory controller has to arbitrate between GPU, CPU and display engine, so here we have that the buffers compete with other traffic
   2. Shared memory is page aligned
   3. *So using instead a private buffer* by doing a blit copy can make it so the kernels can run faster as we don't need CPU cache coherency anymore, paging, etc.
5. Compute unit (a physical core)
   1. apple threadgroup unit
   2. Nvidia SM (and we have read in the programming parallel processors that SM is composed of multiple cuda cores that are the ones that execute the threads, and at least in the architecture they show each SM is composed of two blocks and this two blocks share a cache)
6. A register file is per compute unit (not per thread)
   1. So for example if each thread has 256 registers and there are 32 lanes (threads) and they are of 4 bytes the registers (float32) then we have 256 x 32 x 4 = 32kb
      1. if our threadgroup surpasses this then the registers could spill into global memory
7. FP32 pipes can run two FP16 operations by doing "emulation" or also called "packed math"

### M1 pro
[Apple gpu microarchitecture information](https://github.com/philipturner/metal-benchmarks)

- Apple executes 32 threads in lock-step
  - 
- Comparing to Cuda apple has SIMD not SIMT (as the threads can't diverge, so it is more difficult with threads that access scattered memory addresses)
- The gpu here has 16 cores (A GPU core is analogous to a SM; streaming multiprocessor)
- Each core has 128 scalar ALUs (so we have 16 * 128 = 2048 alus)
- Each ALU is grouped into 4 SIMD units where each SIMD units has 32 threads
  - So like every core has 4 schedulers, so every clock a scheduler can pick a warp and issue one instruction to the SIMD unit (well not SIMD exactly because the threads can work on data that is at random positions)
    - So each core can have 4 warps per cycle
    - We have 3 types of dispatch
      - Quardruple dispatch (4 warps per cycle): here we have perfect register pressure, but this one is not common to have (this is from 4 SIMDs) to not have register file bottleneck
      - Dual dispatch (2 warps per cycle): here the 4 SIMD units are split into two pairs (this is from 2 SIMDs)
      - And single dispatch: this is from 1 SIMD
- A core can have 768 threads? (hardware sweet spot occupancy)
- We have around ~5 TFLOPs (16cores * 128alus * 1.3GHZ * 2 (as FMA counts as 2 FLOPs)
- Max threadgroup size is <= 1024 threads
- It seems each SIMD group runs 64 threads in lock-step
- Register file size is ~208 KB
- Each SIMD (The group of 4 x 32 threads = 128 scalar Alus) has 256 vector registers
  - This way by having more vector registers than threads we can keep many live values without spilling into memory

