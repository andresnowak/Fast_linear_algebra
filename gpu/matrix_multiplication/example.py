import numpy as np
import Metal  # PyObjC wrapper
import objc
from compile_metal import compile_metal_source, remove_file
import os
import ctypes

compile_metal_source("square.metal")

# --------------------------------------------------
# 1. Device & library
# --------------------------------------------------
device = Metal.MTLCreateSystemDefaultDevice()
if device == None:
    raise RuntimeError("Device not found")

metallib_path = os.path.abspath("square.metallib")
lib_url = Metal.NSURL.fileURLWithPath_(metallib_path)

library, err = device.newLibraryWithURL_error_(lib_url, None)
if err:
    raise RuntimeError(err)

kernel = library.newFunctionWithName_("square")

# --------------------------------------------------
# 2. Pipeline state
# --------------------------------------------------
descriptor = Metal.MTLComputePipelineDescriptor.alloc().init()
descriptor.setComputeFunction_(kernel)
pipeline = device.newComputePipelineStateWithDescriptor_error_(descriptor, None)

# --------------------------------------------------  
# Queue
cmd_queue = device.newCommandQueue()
# -------------------------------------------------- 

# --------------------------------------------------
# 3. Data
# --------------------------------------------------
N = 1024
a = np.arange(N, dtype=np.float32)
b = np.empty_like(a)

# Metal buffers (private)
buf_a = device.newBufferWithBytes_length_options_(
    a.tobytes(), a.nbytes, Metal.MTLResourceStorageModeShared
)
buf_b = device.newBufferWithBytes_length_options_(
    b.tobytes(), b.nbytes, Metal.MTLResourceStorageModeShared
)

# Metal buffers (private)
buf_a_d = device.newBufferWithLength_options_(
    a.nbytes, Metal.MTLResourceStorageModePrivate
)
buf_b_d = device.newBufferWithLength_options_(
    b.nbytes, Metal.MTLResourceStorageModePrivate
)

copy_cmd_buffer = cmd_queue.commandBuffer()
copy_enc = copy_cmd_buffer.blitCommandEncoder()

copy_enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(buf_a, 0, buf_a_d, 0, a.nbytes)

copy_enc.endEncoding()
copy_cmd_buffer.commit()
copy_cmd_buffer.waitUntilCompleted()

# --------------------------------------------------
# 4. Command buffer & encoder
# --------------------------------------------------
cmd_buffer = cmd_queue.commandBuffer()
encoder = cmd_buffer.computeCommandEncoder()
encoder.setComputePipelineState_(pipeline)
encoder.setBuffer_offset_atIndex_(buf_a_d, 0, 0)
encoder.setBuffer_offset_atIndex_(buf_b_d, 0, 1)

# Grid size
grid_size = Metal.MTLSizeMake(N, 1, 1)
thread_group_size = Metal.MTLSizeMake(
    min(pipeline.maxTotalThreadsPerThreadgroup(), N), 1, 1
)
encoder.dispatchThreads_threadsPerThreadgroup_(grid_size, thread_group_size)
encoder.endEncoding()

# --------------------------------------------------
# 5. Run & copy back
# --------------------------------------------------
cmd_buffer.commit()
cmd_buffer.waitUntilCompleted()

# --------------------------------------------------
# Copy to back to cpu
# --------------------------------------------------

copy_cmd_buffer = cmd_queue.commandBuffer()
copy_enc = copy_cmd_buffer.blitCommandEncoder()

copy_enc.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
    buf_b_d, 0, buf_b, 0, b.nbytes
)

copy_enc.endEncoding()
copy_cmd_buffer.commit()
copy_cmd_buffer.waitUntilCompleted()

# Map buffer back to numpy
b_np = np.frombuffer(
    buf_b.contents().as_buffer(N * np.dtype(np.float32).itemsize),
    dtype=np.float32,
    count=N,
)
np.copyto(b, b_np)

print("Input :", a[:8])
print("Output:", b[:8])
assert np.allclose(b, a * a)
print("✅ Metal kernel ran successfully from Python!")

remove_file(metallib_path)