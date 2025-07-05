**Apple silicon has 32 ymm registers, 128 bit wide**
**Cache lane size is 64bytes**


## EXTRA
TLDR for the below is that compiler optimizations only get you so far, autotuning is meant to solve this issue but it's being redesigned after many compiler changes.

Full answer you might be interested in, I had a similar line of questioning internally:

question:
I'm seeing big speedups using simdwidthof[type]() * 2  on AVX and simdwidthof[type]() * 4 on apple silicon even in vectorize + parallelize functions. Should we be returning the value of simd_width * the amount of vectors that can be processed in parallel?

answer from one of the kernel engineers:
for(int i=0; i<n; i++) {
    a += b[i];
}

This has a dependency chain. It first has to do a += b[0] and then a += b[1] and so forth at least with floating point. So there is a latency between each addition. It depends on the platform. With Skylake it's 4 cycles. You can unroll the loop to break the dependency. But you can also increase the SIMD size. So if you use twice the natural size (256-bits with AVX) this gives you mostly two independent operations. These can then run in parallel.
So using twice the SIMD size helps break these dependency chains. It fills in latency holes. With Apple it's likely using Neon. It can do 4x128 bit operations in parallel. But only if you break the dependency. So using 4 the SIMD width could be best.

Then I was asking about if simdwidthof should return a platform-specific simd_width_of_type * simd_operations_per_cycle and the answer was:
With matmul for example we would not want to do this. We want the natural size and we use several of them in parallel. It depends on the operations. With the Mandelbrot I use 2* simd_width with hyper-threading and without hyper-threading I would use 4xsimd_width.  This helps fill in the latency holes. It's not just a question of how many can operate in parallel. It's also about filling in the holes so that the pipelines are fully filled.

My question:
So like arithmetic might be different to logical operations?

answer:
Well at least with avx512 you could use up to 8*simd_width to fully saturate the pipelines with fma, multiplication, or addition. You either use 8*simd_width or use 8 different simd variables. Normally you have a mix of different operations. So it's hard to say what will help the most without knowing what is being done

If there is no dependency chain then you don't have to worry about this. What I mean is you may have a mix of dependant and independent operations. The independent operations probably already fill in part of the pipelines. So the factor you use for simd_width will depend on this.

Sorry if I'm too technical. I don't think you can find a single simd_operations_per_cycle value you can use. That's what I'm trying to say.

And someone else added:
It is super algorithm specific as to whether this will be a speedup. This is really why autotuning matters, though we haven't had time to invest in it recently

Getting the absolute max performance is with SIMD and parallel operations is complicated, and isn't something that can easily be done by compiler optimizations like auto vectorization, it only gets you so far. I've been trying to find a way to express this for a blog post in a simple way with real-world examples, but haven't completed it yet.-