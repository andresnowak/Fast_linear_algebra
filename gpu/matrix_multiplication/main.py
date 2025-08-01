import numpy as np
import Metal  # PyObjC wrapper
import objc
from compile_metal import compile_metal_source, remove_file
import os
import struct


def copy_data(cmd_queue, srcs: list, dsts: list, sizes: list) -> float:
    # This can also be done using the resourceManaged instead where instead of copying we use blitencoder to synchronize cpu and gpu
    copy_cmd_buffer = cmd_queue.commandBuffer()

    copy_enc = copy_cmd_buffer.blitCommandEncoder()

    for src, dst, size in zip(srcs, dsts, sizes):
        copy_enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
            src, 0, dst, 0, size
        )

    copy_enc.endEncoding()
    copy_cmd_buffer.commit()
    copy_cmd_buffer.waitUntilCompleted()


def run_kernel(A: np.ndarray, B: np.ndarray, C: np.ndarray, metallib_path: str):
    device = Metal.MTLCreateSystemDefaultDevice()

    lib_url = Metal.NSURL.fileURLWithPath_(metallib_path)

    library, err = device.newLibraryWithURL_error_(lib_url, None)
    if err:
        raise RuntimeError(err)

    kernel = library.newFunctionWithName_("matmul")

    # Queue
    cmd_queue = device.newCommandQueue()

    # Metal buffers (private)
    buf_A = device.newBufferWithBytes_length_options_(
        A.tobytes(), A.nbytes, Metal.MTLResourceStorageModeShared
    )
    buf_B = device.newBufferWithBytes_length_options_(
        B.tobytes(), B.nbytes, Metal.MTLResourceStorageModeShared
    )
    buf_C = device.newBufferWithLength_options_(
        C.nbytes, Metal.MTLResourceStorageModeShared
    )

    # Metal buffers (private)
    buf_A_d = device.newBufferWithLength_options_(
        A.nbytes, Metal.MTLResourceStorageModePrivate
    )
    buf_B_d = device.newBufferWithLength_options_(
        B.nbytes, Metal.MTLResourceStorageModePrivate
    )
    buf_C_d = device.newBufferWithLength_options_(
        C.nbytes, Metal.MTLResourceStorageModePrivate
    )

    copy_data(cmd_queue, [buf_A, buf_B], [buf_A_d, buf_B_d], [A.nbytes, B.nbytes])

    # Compute pipeline
    descriptor = Metal.MTLComputePipelineDescriptor.alloc().init()

    descriptor.setComputeFunction_(kernel)
    pipeline = device.newComputePipelineStateWithDescriptor_error_(descriptor, None)

    cmd_buffer = cmd_queue.commandBuffer()

    encoder = cmd_buffer.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)

    encoder.setBuffer_offset_atIndex_(buf_A_d, 0, 0)
    encoder.setBuffer_offset_atIndex_(buf_B_d, 0, 1)
    encoder.setBuffer_offset_atIndex_(buf_C_d, 0, 2)

    N = A.shape[0]
    K = A.shape[1]
    M = B.shape[1]
    params = struct.pack("III", N, K, M)
    encoder.setBytes_length_atIndex_(params, len(params), 3)

    # print(f"Max threads per threadgroup: {pipeline.maxTotalThreadsPerThreadgroup()}")
    tile_size = 32
    thread_group_size = Metal.MTLSizeMake(tile_size, tile_size, 1)
    grid_size = Metal.MTLSizeMake(np.ceil(M / tile_size) * tile_size, np.ceil(N / tile_size) * tile_size, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
    encoder.endEncoding()

    # Run
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()

    # kernel time
    gpu_start = cmd_buffer.GPUStartTime()  # seconds
    gpu_end = cmd_buffer.GPUEndTime()  # seconds
    kernel_time = (gpu_end - gpu_start) * 1e9  # nanoseconds

    # Copy back to cpu
    copy_data(cmd_queue, [buf_C_d], [buf_C], [C.nbytes])

    # Copy to numpy
    c_np = np.frombuffer(
        buf_C.contents().as_buffer(C.nbytes),
        dtype=np.float32,
        count=C.size,
    ).reshape(N, M)

    np.copyto(C, c_np)

    return kernel_time


def test(A: np.ndarray, B: np.ndarray, C: np.ndarray, metallib_path: str):
    _ = run_kernel(A, B, C, metallib_path)

    print("Output C: ", C[:2, :8])
    print("Output A @ B: ", (A @ B)[:2, :8], "\n")
    try:
        np.testing.assert_allclose(
            C, A @ B, rtol=1e-5, atol=1e-8,
            err_msg="Matrix multiplication mismatch"
        )
    except AssertionError as e:
        product = A @ B
        diff = np.abs(C - product)
        tol = 1e-8 + 1e-5 * np.abs(product)
        # find the first location where |C - A@B| > tol
        i, j = np.argwhere(diff > tol)[0]
        print(
            f"First mismatch at ({i}, {j}): "
            f"C={C[i, j]:.6g}, "
            f"A@B={product[i, j]:.6g}, "
            f"diff={diff[i, j]:.6g}"
        )
        # re-raise so the test still fails
        raise


def benchmark(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    metallib_path: str,
    warmups: int = 3,
    measure_iterations: int = 10,
):
    n, k, m = A.shape[0], A.shape[1], B.shape[1]
    flops = 2 * k * n * m

    # Warm-up
    for _ in range(warmups):
        _ = run_kernel(A, B, C, metallib_path)

    times = []
    for _ in range(measure_iterations):
        times.append(run_kernel(A, B, C, metallib_path))

    best_ns = min(times)  # or np.median(times)
    median_ns = np.median(times)
    mean_ns = np.mean(times)

    gflops = flops / best_ns

    print(f"Best kernel time: {best_ns:.2f} ns  ({best_ns * 1e-6:.3f} ms)")
    print(f"Median kernel time: {median_ns:.2f} ns  ({median_ns * 1e-6:.3f} ms)")
    print(f"Mean kernel time: {mean_ns:.2f} ns  ({mean_ns * 1e-6:.3f} ms)")
    print(f"Best GFLOPS: {gflops:.2f}")


if __name__ == "__main__":
    n = 1024
    m = 1024
    k = 1024

    A = np.random.uniform(0, 5, size=(n, k)).astype(np.float32)
    B = np.random.uniform(0, 5, size=(k, m)).astype(np.float32)
    C = np.zeros((n, m), dtype=np.float32)

    compile_metal_source("matmul.metal")
    compile_metal_source("matmul_tiled.metal")
    compile_metal_source("matmul_tiled_boundary_check.metal")

    metallib_path = os.path.abspath("matmul.metallib")

    test(A, B, C, metallib_path)
    benchmark(A, B, C, metallib_path)
    print()

    metallib_path = os.path.abspath("matmul_tiled.metallib")

    test(A, B, C, metallib_path)
    benchmark(A, B, C, metallib_path)

    metallib_path = os.path.abspath("matmul_tiled_boundary_check.metallib")

    n = 1031
    m = 1033
    k = 1021

    A = np.random.uniform(0, 5, size=(n, k)).astype(np.float32)
    A = np.ones_like(A)
    B = np.random.uniform(0, 5, size=(k, m)).astype(np.float32)
    B = np.ones_like(B)
    C = np.zeros((n, m), dtype=np.float32)

    test(A, B, C, metallib_path)
    benchmark(A, B, C, metallib_path)

    remove_file("matmul.metallib")
    remove_file("matmul_tiled.metallib")
    remove_file("matmul_tiled_boundary_check.metallib")
