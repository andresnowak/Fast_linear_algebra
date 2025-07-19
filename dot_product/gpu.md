Programming massively parallel processors

## CPU
1. So first in a cpu the design is more targeted to single threaded with the idea of the control logic where this allows to execute things in out of order or even in parallel in a single thread.
2. And we also have the data and instruction caches to reduce the latencies of accessing the instruction and data from ram 

## GPU
1. A gpu is more designed to parallel computation
2. A gpu has a bigger bandwith compared to the cpu (this is because of necessity for things like graphic frame buffer requirements)
3. Talking about M1 gpu this one has simd cores (I think 16 cores for M1 pro), then each simd core has simd groups (or warps in nvidia) and each of this simd groups has 32 simd lanes (so each group can execute an instruction in 32 values in parallel. And each lane has its own private register file of scalar values). *So we should have 512 FP32 ops*
4. and each lane has a register file size of 256 32bit regisers?
