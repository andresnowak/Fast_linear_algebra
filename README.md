# Fast Linear Algebra

This repository is a deep dive into optimizing fundamental linear algebra operations. It explores a range of performance engineering techniques on both CPU and GPU.

-   **CPU Optimization**: Written in [Mojo](https://www.modular.com/mojo)
-   **GPU Programming**: Using Apple's [Metal](https://developer.apple.com/metal/) framework (with Objective-C++ and Python).

## Project Structure

```text
├── pixi.toml                         # Mojo project configuration
├── cpu/
│   └── matrix_multiplication/       # GEMM optimization in Mojo
│
└── gpu/
    ├── dot_product/                 # Dot product optimization in Metal/Objective-C++
    └── matrix_multiplication/       # Matrix multiplication in Metal/Python
```

---

## Part 1: CPU Optimization - GEMM in Mojo

This section details the journey of optimizing General Matrix Multiplication (GEMM) on the CPU using Mojo. We start with a textbook implementation and incrementally apply advanced techniques to leverage modern CPU architecture features.

The implementations for each stage can be found in `cpu/matrix-multiplication/`.

### Optimization Stages

1.  **Naive GEMM**: The classic three-loop algorithm.
2.  **Loop Reordering**: Swapping loop order to improve cache locality for one of the input matrices.
3.  **SIMD**: Using Single Instruction, Multiple Data to perform vectorized computations.
4.  **Micro-Kernel**: A highly-optimized core function that computes a small tile of the result matrix, designed to maximize register usage.
5.  **Cache Blocking**: A BLIS-style approach that partitions matrices into blocks to fit into different levels of the CPU cache, reducing trips to main memory.
6.  **Parallelization**: Using Mojo's `parallelize` to distribute the work across multiple CPU cores.

> **For a complete, in-depth explanation of each optimization technique and the theory behind it, please see the detailed write-up:**
>
> ### 📄 [CPU GEMM Optimization Details](./cpu/matrix-multiplication/gemm.md)

### Running the Mojo Code

**Prerequisites:**
-   [Pixi](https://pixi.sh/)

**Setup and Execution:**
`pixi shell`
`mojo run main.mojo`

## Part 2: GPU Optimization - Metal

This section explores GPU programming using Apple's Metal framework. It includes implementations for both dot product (in Objective-C++) and matrix multiplication (in Python).

### Dot Product in Objective-C++ and Metal

This part focuses on a common parallel programming challenge: the reduction operation. We implement a dot product using several different reduction strategies on the GPU.

**Reduction Strategies Explored:**
-   GPU multiplication with final reduction on the CPU.
-   Partial reduction on the GPU within threadgroups.
-   Full reduction on the GPU using atomic operations.
-   Tree-based reduction within a threadgroup to minimize divergence and maximize parallelism.
-   Hierarchical reduction to handle inputs larger than a single threadgroup.

> **For a detailed discussion of the algorithms, GPU architecture concepts, and trade-offs, refer to the documentation:**
>
> ### 📄 [GPU Dot Product Optimization Details](./gpu/dot_product/dot_product.md)

### Running the Dot Product Code

**Prerequisites:**
-   macOS with Xcode Command Line Tools installed.

**Build and Execution:**
1.  Navigate to the dot product directory:
    ```bash
    cd gpu/dot_product
    ```
2.  Build the project using the Makefile and run:
    ```bash
    make run
    ```

### Matrix Multiplication in Python and Metal

This implementation provides a baseline for GEMM on the GPU, using Python and the [PyObjC](https://pyobjc.readthedocs.io/en/latest/) bridge to interface with Metal. The kernel uses a simple approach where each thread in the compute grid is responsible for calculating a single element of the output matrix C.

> **For a brief overview of the Metal kernel and Python setup, see:**
>
> ### 📄 [GPU Matrix Multiplication Details](./gpu/matrix_multiplication/matmul.md)

### Running the Matrix Multiplication Code

**Prerequisites:**
-   macOS with a Metal-capable GPU.
-   Python 3.12+
-   A Python package manager like `pip` or `uv`.

**Setup and Execution:**
1.  Navigate to the matrix multiplication directory:
    ```bash
    cd gpu/matrix_multiplication
    ```
2.  Install the required Python dependencies from `pyproject.toml`.
    ```bash
    # Using uv
    uv lock
    ```
3.  Run the script. It will compile the Metal kernel, run the matrix multiplication, and verify the result against NumPy.
    ```bash
    python main.py
    ```