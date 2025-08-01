import numpy as np
import Metal  # PyObjC wrapper
import objc
from compile_metal import compile_metal_source, remove_file
import os
import struct


def copy_data(cmd_queue, srcs: list, dsts: list, sizes: list):
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

    print(pipeline.maxTotalThreadsPerThreadgroup())
    grid_size = Metal.MTLSizeMake(M, N, 1)
    thread_group_size = Metal.MTLSizeMake(
        32, 32, 1
    )
    encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
    encoder.endEncoding()

    # Run
    cmd_buffer.commit()
    cmd_buffer.waitUntilCompleted()

    # Copy back to cpu
    copy_data(cmd_queue, [buf_C_d], [buf_C], [C.nbytes])

    # Copy to numpy
    c_np = np.frombuffer(
        buf_C.contents().as_buffer(C.nbytes),
        dtype=np.float32,
        count=C.size,
    ).reshape(N, M)

    np.copyto(C, c_np)


if __name__ == "__main__":
    n = 1024
    m = 1024
    k = 1024

    A = np.random.uniform(0, 5, size=(n, k)).astype(np.float32)
    B = np.random.uniform(0, 5, size=(k, m)).astype(np.float32)
    C = np.zeros((n, m), dtype=np.float32)

    compile_metal_source("matmul.metal")

    metallib_path = os.path.abspath("matmul.metallib")

    run_kernel(A, B, C, metallib_path)

    print("Output C: ", C)
    print("Output A @ B: ", (A @ B))
    assert np.allclose(A @ B, C)

    remove_file("matmul.metallib")
