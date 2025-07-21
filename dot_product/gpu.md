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



3. Talking about M1 gpu this one has simd cores (I think 16 cores for M1 pro), then each simd core has simd groups (or warps in nvidia) and each of this simd groups has 32 simd lanes (so each group can execute an instruction in 32 values in parallel. And each lane has its own private register file of scalar values). *So we should have 512 FP32 ops*
4. and each lane has a register file size of 256 32bit registers?

### M1 pro
[https://github.com/philipturner/metal-benchmarks](Apple gpu microarchitecture information)

- The gpu here has 16 cores (A GPU core is analogous to a SM; streaming multiprocessor)
- Each core has 128 scalar ALUs
- Each ALU is grouped into 4 SIMD units where each SIMD units has 32 threads
  - So like every core has 4 schedulers, so every clock a scheduler can pick a warp and issue one instruction to the SIMD unit (well not SIMD exactly because the threads can work on data that is at random positions)
    - So each core can have 4 warps per cycle
    - We have 3 types of dispatch
      - Quardruple dispatch (4 warps per cycle): here we have perfect register pressure, but this one is not common to have (this is from 4 SIMDs) to not have register file bottleneck
      - Dual dispatch (2 warps per cycle): here the 4 SIMD units are split into two pairs (this is from 2 SIMDs)
      - And single dispatch: this is from 1 SIMD
  - A core can have 768 threads? (hardware sweet spot occupancy)
